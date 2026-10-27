import AcceleratedKernels as AK
import KernelAbstractions as KA
using LinearAlgebra: Symmetric, dot, eigmin, eigvals, tr
using StaticArrays: SMatrix, SVector

export SurfaceTerm, surface_term!, depolarization_tensor

# ---------------------------------------------------------------------------
#  SurfaceTerm — depolarization-field ("Ewald surface") term for the k = 0
#  polarisation mode, generalised to an anisotropic external medium and an
#  ellipsoidal embedding shape (Issue #179).
#
#      U_surf = (2π/V) δM · T_Ω · δM,          δM = M(t) − M_ref(t)
#
#  with the real-polarisation depolarization tensor
#
#      T_Ω = [I + L_Ω (ε_in − ε_out)]⁻¹ L_Ω,   ε_in = I
#
#      L_Ω = (1/4π) ∫_{S²} (A⁻¹n̂)⊗(A⁻¹n̂) / (A⁻¹n̂ · ε_out · A⁻¹n̂) dω
#
#  ε_in = I because the explicit MD charges interact through the bare
#  Coulomb kernel — the cell's own dielectric response is carried by the
#  charges themselves, not by a background permittivity.
#
#  The reciprocal-space Coulomb solvers (Ewald / PME) implement the
#  conductor ("tin-foil") boundary, T → 0; this term is exactly what a
#  finite anisotropic external permittivity adds on top, so its absence
#  (no :surface_term key) IS the conductor case.  Everything about T_Ω is
#  decided at construction (a one-time spherical quadrature); the per-step
#  work is one whole-system dipole reduction and a uniform per-charge
#  force — O(N), no FFT, no groups.
#
#  Conventions (pinned in docs/src/theory/surface-term.md):
#  - T_Ω is a LAB-FRAME constant: the embedding shape and the external
#    permittivity describe the world outside the cell and do not co-rotate
#    or co-deform with the lattice.
#  - M_ref(t) = Lᵀ(t)·M_frac_ref co-transforms with the cell (same gauge
#    machinery as the polarisation umbrellas), so pure cell deformation
#    with atoms static in fractional space keeps δM = 0.
# ---------------------------------------------------------------------------

const _SURF_EYE = SMatrix{3,3,Float64}(1, 0, 0, 0, 1, 0, 0, 0, 1)

# --- input normalisation ---------------------------------------------------

# ε_out: scalar (isotropic), 3-vector (diagonal), or full 3×3 symmetric
# positive-definite matrix.  Always returns an SMatrix{3,3,Float64}.
_surf_epsilon(x::Real) = begin
    x > 0 || error("SurfaceTerm: eps_out must be > 0 (got $x)")
    Float64(x) * _SURF_EYE
end
_surf_epsilon(x::AbstractVector) = begin
    length(x) == 3 ||
        error("SurfaceTerm: a vector eps_out must have 3 entries " *
              "(the diagonal), got $(length(x))")
    all(>(0), x) || error("SurfaceTerm: eps_out diagonal must be > 0 (got $x)")
    SMatrix{3,3,Float64}(x[1], 0, 0, 0, x[2], 0, 0, 0, x[3])
end
_surf_epsilon(x::AbstractMatrix) = _surf_spd(x, "eps_out")

# Shape: :sphere, a 3-vector of semiaxes, or the full symmetric
# positive-definite matrix A mapping the unit ball onto the ellipsoid
# Ω = {A·x : |x| < 1} (a rotated ellipsoid is A = R·diag(a)·Rᵀ).  The
# overall scale of A cancels in L_Ω, so semiaxes need no normalisation.
_surf_shape(s::Symbol) = begin
    s === :sphere ||
        error("SurfaceTerm: unknown shape $(repr(s)) — pass :sphere, a " *
              "3-vector of semiaxes, or a 3×3 symmetric positive-definite " *
              "matrix A (unit ball → ellipsoid)")
    _SURF_EYE
end
_surf_shape(s::AbstractVector) = begin
    length(s) == 3 ||
        error("SurfaceTerm: a vector shape must have 3 semiaxes, got " *
              "$(length(s))")
    all(>(0), s) || error("SurfaceTerm: semiaxes must be > 0 (got $s)")
    SMatrix{3,3,Float64}(s[1], 0, 0, 0, s[2], 0, 0, 0, s[3])
end
_surf_shape(s::AbstractMatrix) = _surf_spd(s, "shape")

function _surf_spd(x::AbstractMatrix, name::String)
    size(x) == (3, 3) ||
        error("SurfaceTerm: $name must be 3×3 (got $(size(x)))")
    M = SMatrix{3,3,Float64}(x)
    maximum(abs, M - M') <= 1e-10 * max(1.0, maximum(abs, M)) ||
        error("SurfaceTerm: $name must be symmetric (max |x - xᵀ| = " *
              "$(maximum(abs, M - M'))) — symmetrise it explicitly if the " *
              "asymmetry is numerical noise")
    Msym = (M + M') / 2
    λ = eigmin(Symmetric(Matrix(Msym)))
    λ > 0 || error("SurfaceTerm: $name must be positive definite " *
                   "(eigmin = $λ)")
    return Msym
end

# --- depolarization tensor -------------------------------------------------

# One Fibonacci-sphere pass: L ≈ mean over near-uniform directions n̂ of
# (A⁻¹n̂)⊗(A⁻¹n̂)/(A⁻¹n̂·ε·A⁻¹n̂).  The mean IS the (1/4π)∫dω because the
# k-integral of the note separates into this angular part times a radial
# part that Parseval fixes to 1/4π (docs/src/theory/surface-term.md).
function _surf_L_quad(ε::SMatrix{3,3,Float64}, G::SMatrix{3,3,Float64}, n::Int)
    acc = zero(SMatrix{3,3,Float64,9})
    ga = π * (3.0 - sqrt(5.0))
    for i in 0:(n - 1)
        t = i + 0.5
        z = 1.0 - 2.0 * t / n
        r = sqrt(max(0.0, 1.0 - z * z))
        ϕ = ga * t
        v = G * SVector(r * cos(ϕ), r * sin(ϕ), z)
        acc += (v * v') / dot(v, ε * v)
    end
    L = acc / n
    return (L + L') / 2
end

"""
    depolarization_tensor(eps_out; shape = :sphere, rtol = 1e-8,
                          n0 = 40_000, nmax = 10_240_000) -> Matrix{Float64}

Depolarization tensor `L_Ω` of a uniformly polarised ellipsoid embedded in
an infinite anisotropic dielectric `eps_out`:

```
L_Ω = (1/4π) ∫_{S²} (A⁻¹n̂)⊗(A⁻¹n̂) / (A⁻¹n̂ · ε_out · A⁻¹n̂) dω
```

`eps_out` is a scalar, the 3 diagonal entries, or a full 3×3 symmetric
positive-definite matrix. `shape` is `:sphere`, a 3-vector of semiaxes, or
the symmetric positive-definite matrix `A` mapping the unit ball onto the
ellipsoid (overall scale cancels). Isotropic sphere: `L = I/(3ε)`. The
exact identity `Tr(ε_out·L) = 1` holds for every shape.

Evaluated by Fibonacci-sphere quadrature, doubling the direction count from
`n0` until the componentwise change drops below `rtol` (relative to
`max(1, |L|)`; the components of `L` are ≤ 1, so the floor makes this
effectively an *absolute* tolerance of `rtol`). Errors if `nmax` is reached
without converging. The identity `Tr(ε_out·L) = 1` is asserted afterwards,
but note it holds **identically** for this direction-average form — each
direction contributes `(v·εv)/(v·εv)` — so it pins the normalisation
contract (a dropped `1/n`, a wrong denominator) and cannot detect an
unconverged quadrature; convergence rests on the doubling loop, and the
independent accuracy evidence is the closed-form comparisons in
`test/calculators/test_surface_term.jl`.
"""
function depolarization_tensor(eps_out; shape = :sphere, rtol::Real = 1e-8,
                               n0::Integer = 40_000,
                               nmax::Integer = 10_240_000)
    ε = _surf_epsilon(eps_out)
    A = _surf_shape(shape)
    G = inv(A)
    (n0 >= 100 && nmax > 2 * n0) ||
        error("depolarization_tensor: need n0 >= 100 and nmax > 2·n0 " *
              "(got n0 = $n0, nmax = $nmax)")

    n = Int(n0)
    L_prev = _surf_L_quad(ε, G, n)
    L = L_prev
    converged = false
    while 2n <= Int(nmax)
        n *= 2
        L = _surf_L_quad(ε, G, n)
        if maximum(abs, L - L_prev) <= rtol * max(1.0, maximum(abs, L))
            converged = true
            break
        end
        L_prev = L
    end
    converged ||
        error("depolarization_tensor: quadrature did not converge to " *
              "rtol = $rtol within $n directions.  Raise nmax or loosen " *
              "rtol — refusing to hand back an unconverged tensor")

    # Normalisation contract only: the identity holds for ANY direction
    # set (each term contributes (v·εv)/(v·εv) = 1/n), so this catches a
    # dropped 1/n or a wrong denominator — an unconverged quadrature
    # passes it.  Convergence is the doubling loop's job above; accuracy
    # is pinned independently by the closed-form tests.
    trerr = abs(tr(ε * L) - 1)
    trerr <= 1e-6 ||
        error("depolarization_tensor: Tr(ε_out·L) = $(tr(ε * L)) violates " *
              "the identity Tr = 1 by $trerr — the quadrature " *
              "normalisation is broken, refusing to continue")
    return Matrix(L)
end

# Real-polarisation depolarization tensor for ε_in = I:
#   T = [I + L(I − ε_out)]⁻¹ L
# Analytically T⁻¹ = L⁻¹ + (I − ε_out) is symmetric, so any asymmetry is a
# numerical defect; positive definiteness is the physical requirement
# U_surf ≥ 0 (a depolarization field costs energy), so a violation means
# the inputs are outside the theory's domain — stop, do not clamp.
function _surf_T_from_L(L::AbstractMatrix, ε::AbstractMatrix)
    E3 = Matrix(_SURF_EYE)
    T = (E3 + Matrix(L) * (E3 - Matrix(ε))) \ Matrix(L)
    asym = maximum(abs, T - T')
    asym <= 1e-8 * max(1.0, maximum(abs, T)) ||
        error("SurfaceTerm: T came out asymmetric (max |T - Tᵀ| = $asym) — " *
              "T⁻¹ = L⁻¹ + (I − ε_out) is symmetric analytically, so this " *
              "is a numerical defect in L")
    Tsym = (T + T') / 2
    λ = eigmin(Symmetric(Tsym))
    λ > 0 ||
        error("SurfaceTerm: T is not positive definite (eigmin = $λ); " *
              "U_surf = (2π/V)·δM·T·δM would be negative for some δM — " *
              "refusing this (eps_out, shape) combination")
    return Tsym
end

# --- gauge and neutrality checks (construction time) -----------------------

# The whole-system dipole of a charged system depends on the cell origin, so
# U_surf would change under a rigid translation — the total-dipole analogue
# of the per-group neutrality PolarizationGauge enforces, with the same
# accumulation-scaled tolerance.
function _surf_assert_neutral(atoms::Atoms, qc_table, qs_table)
    elemid = Vector{Int}(Array(atoms.arrays[:elemid]))
    qc = Array(qc_table)
    qs = Array(qs_table)
    net = zero(FPPREC)
    scale = zero(FPPREC)
    for e in elemid
        q = FPPREC(qc[e]) + FPPREC(qs[e])
        net += q
        scale += abs(q)
    end
    tol = 8 * length(elemid) * eps(FPPREC) * max(scale, one(FPPREC))
    abs(net) <= tol ||
        error("SurfaceTerm: the system carries a net charge of $net " *
              "(tolerance $tol).  A charged system's dipole depends on the " *
              "cell origin, so the surface term would change under a rigid " *
              "translation of the system")
    return nothing
end

# Total fractional reference dipole from the declared gauge.  A prepared
# PolarizationGauge brings the full identity/branch validation with it; a
# raw Atoms frame ("this frame is P = 0") gets the total-dipole subset of
# those checks — it has no group structure to compare, but the same-system
# pairing by :id and the half-period branch guard apply unchanged.
function _surf_gauge_mref(gauge::PolarizationGauge, atoms::Atoms,
                          qc_table, qs_table)
    validate_gauge(gauge, atoms)
    return total_mfrac_ref(gauge)
end

function _surf_gauge_mref(ref::Atoms, atoms::Atoms, qc_table, qs_table)
    Cores.domain(ref) isa Cores.MPIDomain &&
        error("SurfaceTerm: the gauge frame must be the full replicated " *
              "frame, not a distributed one")
    n = length(atoms)
    length(ref) == n ||
        error("SurfaceTerm: gauge frame has $(length(ref)) atoms but the " *
              "run frame has $n — it is a different system")
    (haskey(ref.arrays, :id) && haskey(atoms.arrays, :id)) ||
        error("SurfaceTerm: both the gauge frame and the run frame need " *
              ":id — rows are not comparable (a frame from the MD driver " *
              "comes back cell-sorted)")

    ids = Vector{Int}(Array(atoms.arrays[:id]))
    row_of = Dict{Int,Int}(id => i for (i, id) in enumerate(ids))
    id_ref = Vector{Int}(Array(ref.arrays[:id]))
    perm = Vector{Int}(undef, n)
    for (g, id) in enumerate(id_ref)
        r = get(row_of, id, 0)
        r == 0 && error("SurfaceTerm: the run frame has no atom with :id " *
                        "$id, which the gauge frame does — it is a " *
                        "different system")
        perm[g] = r
    end
    Vector{Int}(Array(atoms.arrays[:elemid]))[perm] ==
        Vector{Int}(Array(ref.arrays[:elemid])) ||
        error("SurfaceTerm: :elemid differs from the gauge frame " *
              "(compared per :id)")

    ds = abs.(Array(atoms.arrays[:pos])[perm, :] .- Array(ref.arrays[:pos]))
    dmax = maximum(ds)
    dmax <= 0.45 ||
        error("SurfaceTerm: an atom moved $(round(dmax, digits = 3)) of a " *
              "supercell period relative to the gauge frame.  Past half a " *
              "period the periodic image is ambiguous, so δM would be off " *
              "by a polarisation quantum.  Either the frame is not the " *
              "structure this gauge describes, or :pos was wrapped " *
              "somewhere (it must not be — see CLAUDE.md)")

    return total_mfrac_ref(ref, (; qc = qc_table, qs = qs_table))
end

# --- the term itself -------------------------------------------------------

"""
    SurfaceTerm

Depolarization-field bias on the whole-system dipole moment:

```
U_surf = (2π/V) · δM · T_Ω · δM,    δM = M(t) − Lᵀ(t)·M_frac_ref
```

where `T_Ω` is the real-polarisation depolarization tensor of an ellipsoid
embedded in an anisotropic external dielectric (see
[`depolarization_tensor`](@ref) and docs/src/theory/surface-term.md). This
is the anisotropic generalisation of the Ewald surface term; the isotropic
sphere reduces to the textbook `2π/((2ε_out + 1)V)·|δM|²`. Omitting the
term altogether is the conductor (tin-foil) boundary the Ewald/PME solvers
already implement.

# Fields

- `T` — 3×3 real-polarisation depolarization tensor (lab frame, symmetric
  positive definite; validated at construction).
- `L` — the underlying depolarization tensor `L_Ω` (diagnostic).
- `eps_out`, `shape` — the normalised inputs `T` was built from
  (3×3 each; kept so a run's boundary condition is reconstructable).
- `M_frac_ref` — total fractional dipole of the gauge frame; multiplied by
  `Lᵀ(t)` each step this is the cell-covariant zero-point of `δM`.
- `results` — `"bias"` (`U_surf`) and `"dM"` (`δM`), refreshed on every
  [`surface_term!`](@ref) call.

Build via the constructor; pass it as `params[:surface_term]` to
`FMD.app[:NPTMD]` / [`npt!`](@ref FMD.Apps.npt!).
"""
struct SurfaceTerm
    T::Matrix{FPPREC}
    L::Matrix{FPPREC}
    eps_out::Matrix{FPPREC}
    shape::Matrix{FPPREC}
    M_frac_ref::Vector{FPPREC}
    results::Dict{String,Any}
end

"""
    SurfaceTerm(atoms::Atoms, qc_table, qs_table;
                eps_out, shape = :sphere, gauge,
                quad_rtol = 1e-8) -> SurfaceTerm

Build a depolarization-field term for `atoms`. `qc_table`, `qs_table` are
the per-element core/shell charge tables (the same objects later passed as
`params[:qc]` / `params[:qs]`). With a prepared [`PolarizationGauge`](@ref)
they are used only for the whole-system neutrality check — the gauge's
`M_frac_ref` already encodes the charges it was built with, so build the
gauge and run the MD with the same tables (this is a convention, not
enforced; a mismatch silently shifts the δM zero-point).

- `eps_out` — external relative permittivity: scalar, diagonal 3-vector,
  or full 3×3 symmetric positive-definite tensor (lab frame).
- `shape` — embedding region: `:sphere` (default), a 3-vector of ellipsoid
  semiaxes, or the symmetric positive-definite matrix `A` mapping the unit
  ball onto the ellipsoid (overall scale cancels).
- `gauge` — **required**: the declared `P = 0` reference, either a
  [`PolarizationGauge`](@ref) (normally shared with the polarisation
  umbrellas via [`load_gauge`](@ref)) or an `Atoms` reference frame. There
  is no default: which structure has zero polarisation is physics, and
  `δM` — hence the entire energy — is measured from it.
- `quad_rtol` — convergence target of the one-time spherical quadrature.

Errors on a distributed frame (construct BEFORE `distribute`, like every
gauge consumer), on a charged system (its dipole is origin-dependent), on
a gauge that does not match `atoms`, and on any `(eps_out, shape)` for
which the computed `T` fails its exact identities (see
[`depolarization_tensor`](@ref)).
"""
function SurfaceTerm(atoms::Atoms, qc_table, qs_table;
                     eps_out, shape = :sphere,
                     gauge::Union{PolarizationGauge,Atoms},
                     quad_rtol::Real = 1e-8)
    Cores.domain(atoms) isa Cores.MPIDomain &&
        error("SurfaceTerm: construct from the full replicated frame " *
              "BEFORE distribute (the gauge reference is a whole-system " *
              "quantity)")

    ε = _surf_epsilon(eps_out)
    A = _surf_shape(shape)
    L = depolarization_tensor(ε; shape = A, rtol = quad_rtol)
    T = _surf_T_from_L(L, Matrix(ε))

    _surf_assert_neutral(atoms, qc_table, qs_table)
    mref = _surf_gauge_mref(gauge, atoms, qc_table, qs_table)

    return SurfaceTerm(Matrix{FPPREC}(T), Matrix{FPPREC}(L),
                       Matrix{FPPREC}(Matrix(ε)), Matrix{FPPREC}(Matrix(A)),
                       Vector{FPPREC}(mref), Dict{String,Any}())
end

"""
    surface_term!(atoms::Atoms, st::SurfaceTerm, prms)

Add the depolarization-field energy, forces and virial of `st` to `atoms`:

```
U     = (2π/V) δM·T·δM
F_i   = −q_i · g            (cores and shells, g = (4π/V)·T·δM)
W     = U·I − sym(g ⊗ δM)   (Voigt-6, −∂U/∂ε convention)
```

Called from [`shell_forces!`](@ref) / [`core_forces!`](@ref) when
`prms[:surface_term]` is present — inside the shell CG loop on purpose:
the shells relaxing against the depolarization field is the dielectric
screening this term exists to model. Under domain decomposition the
whole-system dipole is Allreduced (via [`calcdipole!`](@ref)) and every
derived quantity is replicated, so the energy/virial are added directly.
Populates `st.results["bias"]` and `st.results["dM"]`.
"""
function surface_term!(atoms::Atoms, st::SurfaceTerm, prms)
    # Whole-system dipole; calcdipole! Allreduces the owned partial under
    # DD, so everything derived below is replicated on every rank.
    M_abs = Vector{FPPREC}(Array(calcdipole!(atoms, prms)))
    lat = Matrix{FPPREC}(Array(atoms.info[:lattice]))
    # Fractional → Cartesian for a column vector is transpose(L)·m, the
    # counterpart of the row-wise pos·lattice (cf. calc_dipole_phys!).
    M_ref = transpose(lat) * st.M_frac_ref
    dM = M_abs .- M_ref
    V = FPPREC(volume(atoms))
    TdM = st.T * dM
    U = FPPREC(2π) / V * dot(dM, TdM)
    g1, g2, g3 = (FPPREC(4π) / V) .* TdM        # g = dU/dM

    # dM_γ/dx_iα = q_i·δ_γα → a uniform per-charge force −q·g on cores and
    # shells alike (an ExternalField with E = −g, but with the energy and
    # virial this induced field actually carries).
    if length(atoms) > 0
        fcore = atoms.arrays[:forces]
        fshell = atoms.arrays[:shellforces]
        elemid = atoms.arrays[:elemid]
        qc_t = prms[:qc]
        qs_t = prms[:qs]
        AK.foreachindex(elemid) do i
            si = elemid[i]
            fcore[i, 1] -= qc_t[si] * g1
            fcore[i, 2] -= qc_t[si] * g2
            fcore[i, 3] -= qc_t[si] * g3
            fshell[i, 1] -= qs_t[si] * g1
            fshell[i, 2] -= qs_t[si] * g2
            fshell[i, 3] -= qs_t[si] * g3
        end
    end

    atoms.info[:potential_energy] .+= U

    # Closed-form virial (derivation: docs/src/theory/surface-term.md).
    # −sym(g⊗δM) is the atom-loop Σ r⊗F (which telescopes to −g⊗M because
    # the force is uniform per charge) plus the M_ref cell-covariance term
    # +sym(g⊗M_ref); U·I is the explicit 1/V derivative.  T itself is a
    # lab-frame constant and contributes no cell derivative.  Replicated
    # (built from the Allreduced dipole) → added directly, NOT through
    # add_reduced_ev!.
    v6 = FPPREC[U - g1 * dM[1],
                U - g2 * dM[2],
                U - g3 * dM[3],
                -(g2 * dM[3] + g3 * dM[2]) / 2,
                -(g1 * dM[3] + g3 * dM[1]) / 2,
                -(g1 * dM[2] + g2 * dM[1]) / 2]
    vir = atoms.info[:virial]
    backend = KA.get_backend(vir)
    if backend isa KA.CPU
        vir .+= v6
    else
        vir .+= _devcopy(backend, v6)
    end

    st.results["bias"] = U
    st.results["dM"] = dM
    return nothing
end

# --- analytic Hessian-vector product (Issue #181) --------------------------

"""
    SurfaceHvpCache

Frozen-snapshot state of the surface term for [`surface_hvp!`](@ref):
`δM`, the field `g = (4π/V)T·δM` and the prefactor `(4π/V)T` at the
configuration [`hvp_workspace`](@ref) was built for, the per-atom charges
they were built with, and the `(N, 3)` scratch the direction reduction
accumulates into.

Held by the HVP workspace, never rebuilt on its own: `hvp!`'s
frozen-snapshot guard (positions, shell displacements, lattice) is what
keeps `δM` and `V` in step with the atoms.

The charges are part of the snapshot rather than re-read per product (the
same choice the reciprocal caches make): `δM` was built from them, so a
`prms` whose charge tables changed underneath would pair a frozen `δM`
with live charges — half of one configuration and half of another.
"""
struct SurfaceHvpCache{TV<:AbstractVector,TM<:AbstractMatrix}
    st::SurfaceTerm
    dM::Vector{FPPREC}       # δM at the snapshot (replicated under DD)
    g::Vector{FPPREC}        # (4π/V)·T·δM = ∂U/∂M
    pref::Matrix{FPPREC}     # (4π/V)·T
    qc::TV                   # (N,) per-atom core charge
    qs::TV                   # (N,) per-atom shell charge
    scratch::TM              # (N, 3) sink for q_s·v before the reduction
end

"""
    surface_hvp_cache(st::SurfaceTerm, atoms::Atoms, prms) -> SurfaceHvpCache

Freeze the surface term at the current configuration. **Collective**
under domain decomposition (the dipole is Allreduced), so every rank —
empty ones included — must reach it.
"""
function surface_hvp_cache(st::SurfaceTerm, atoms::Atoms, prms)
    # δM is recomputed rather than read back from `st.results` (which the
    # last surface_term! call left there, typically at this very
    # configuration): the results Dict carries no record of WHICH
    # configuration produced it, so reusing it would make the cache
    # silently wrong whenever the caller's evaluation order changes.
    # The cost is one extra dipole reduction per workspace, not per
    # product.
    backend = KA.get_backend(atoms)
    N = length(atoms)
    lat = Matrix{FPPREC}(Array(atoms.info[:lattice]))
    M_abs = Vector{FPPREC}(calcdipole(atoms, prms))
    dM = M_abs .- transpose(lat) * st.M_frac_ref
    pref = (FPPREC(4π) / FPPREC(volume(atoms))) .* st.T
    qc = KA.zeros(backend, FPPREC, N)
    qs = KA.zeros(backend, FPPREC, N)
    if N > 0
        qc_t = prms[:qc]
        qs_t = prms[:qs]
        AK.map!(e -> qc_t[e], qc, atoms.arrays[:elemid])
        AK.map!(e -> qs_t[e], qs, atoms.arrays[:elemid])
    end
    return SurfaceHvpCache(st, dM, pref * dM, pref, qc, qs,
                           KA.zeros(backend, FPPREC, N, 3))
end

# q_s-weighted direction, ready for the column sum.
function _surf_hvp_weight!(sc, v, qs)
    AK.foreachindex(qs) do i
        q = qs[i]
        sc[i, 1] = q * v[i, 1]
        sc[i, 2] = q * v[i, 2]
        sc[i, 3] = q * v[i, 3]
    end
    return sc
end

# Uniform per-charge scatter of the field derivative c = D_v g. Two
# methods rather than a branch inside the device closure: the caller's
# `Kv === nothing` fast path stays statically typed (the CG inner loop
# calls this every iteration).
function _surf_hvp_scatter!(Jv, Kv::AbstractMatrix, qc, qs, c1, c2, c3)
    AK.foreachindex(qs) do i
        a = qs[i]
        b = qc[i]
        Jv[i, 1] -= a * c1
        Jv[i, 2] -= a * c2
        Jv[i, 3] -= a * c3
        Kv[i, 1] -= b * c1
        Kv[i, 2] -= b * c2
        Kv[i, 3] -= b * c3
    end
    return nothing
end

function _surf_hvp_scatter!(Jv, ::Nothing, qc, qs, c1, c2, c3)
    AK.foreachindex(qs) do i
        a = qs[i]
        Jv[i, 1] -= a * c1
        Jv[i, 2] -= a * c2
        Jv[i, 3] -= a * c3
    end
    return nothing
end

"""
    surface_hvp!(Jv, Kv, v, atoms, cache::SurfaceHvpCache) -> Vector

Accumulate the surface term's Hessian blocks along `v` into `Jv` (and
`Kv`, unless `nothing`), and return its Voigt-6 virial derivative
`D_v W` as a host vector.

`U` depends on the positions through the whole-system dipole alone, so
with `S = Σ_j q^s_j v_j` (one 3-vector — the only reduction) the blocks
are rank-3 closed forms:

```
c    = D_v g = (4π/V)·T·S
Jv_i = -(∂²U/∂x_s∂x_s · v)_i = -q^s_i·c
Kv_i = -(∂²U/∂x_c∂x_s · v)_i = -q^c_i·c
```

and, differentiating `surface_term!`'s closed-form virial
`W = U·I - sym(g⊗δM)` along `v`,

```
D_v W = (g·S)·I - sym(c⊗δM) - sym(g⊗S).
```

`D_v W` is a **whole-system** quantity with no per-atom attribution — the
same situation as the reciprocal-space grid virial — so it is the return
value rather than a row of `hvp!`'s `Wv`, and under domain decomposition
it is already replicated (built from the Allreduced `S` and the cache's
Allreduced `δM`) and must **not** be summed again. The `peratomstress`
columns therefore omit it, exactly as they omit the reciprocal part.

The charges come from the cache, not from a live `prms`: see
[`SurfaceHvpCache`](@ref).

**Collective**: the `S` reduction is one Allreduce, so every rank must
call this — including ranks with no owned atoms, which contribute a zero
partial sum but still ride the collective.
"""
function surface_hvp!(Jv, Kv, v, atoms::Atoms, cache::SurfaceHvpCache)
    N = length(atoms)

    # S = Σ_j q^s_j v_j: owned partial sum, then whole-system on every rank.
    S = zeros(FPPREC, 3)
    if N > 0
        sc = _surf_hvp_weight!(cache.scratch, v, cache.qs)
        S .= FPPREC.(Array(AK.sum(sc, dims = 1))[1, :])
    end
    dom = Cores.domain(atoms)
    dom isa Cores.MPIDomain && Cores.allreduce_sum!(dom, S)

    c = cache.pref * S          # D_v g
    N > 0 && _surf_hvp_scatter!(Jv, Kv, cache.qc, cache.qs, c[1], c[2], c[3])

    g = cache.g
    dM = cache.dM
    dU = g[1] * S[1] + g[2] * S[2] + g[3] * S[3]
    return FPPREC[dU - c[1] * dM[1] - g[1] * S[1],
                  dU - c[2] * dM[2] - g[2] * S[2],
                  dU - c[3] * dM[3] - g[3] * S[3],
                  -(c[2] * dM[3] + c[3] * dM[2]) / 2 -
                      (g[2] * S[3] + g[3] * S[2]) / 2,
                  -(c[1] * dM[3] + c[3] * dM[1]) / 2 -
                      (g[1] * S[3] + g[3] * S[1]) / 2,
                  -(c[1] * dM[2] + c[2] * dM[1]) / 2 -
                      (g[1] * S[2] + g[2] * S[1]) / 2]
end

"""
    surface_spectral_bump(st::SurfaceTerm, atoms::Atoms, prms) -> FPPREC

Largest eigenvalue the surface term adds to the preconditioned shell
operator `A⁻¹H_ss` of shadow XL-BOMD (`A` = spring Hessian):

```
Δλ = (4π/V)·(Σ_i q^s_i²/k2_i)·λmax(T_Ω)
```

Exact, not a bound: with `w_i = q^s_i/√k2_i` the symmetrised block
`A^{-1/2}H_surf A^{-1/2}` is `(4π/V)·(w wᵀ)⊗T`, whose non-zero
eigenvalues are `(4π/V)|w|²·λ_k(T)`. Since `H_ss = H_rest + H_surf` with
`H_rest` positive semi-definite at a stable shell configuration, Weyl's
inequality makes this a **lower bound on the full `λmax`** — which is
what lets [`xl_init!`](@ref FMD.Apps.xl_init!) refuse a declared spectral
interval that cannot contain it (`:chebyshev`) or an `xl_kappa` that is
provably unstable (`:spring` only — the Chebyshev kernel's coupling is
the flattened `λ·p(λ)`, not `λ`; theory note `xlbomd-shell.md` §6 and
§6.1).

Species with `k2 = 0` carry no shell degree of freedom and are projected
out of that operator, so they are skipped here too.
"""
function surface_spectral_bump(st::SurfaceTerm, atoms::Atoms, prms)
    elemid = Vector{Int}(Array(atoms.arrays[:elemid]))
    qs = Vector{FPPREC}(Array(prms[:qs]))
    k2 = Vector{FPPREC}(Array(prms[:k2]))
    w2 = zero(FPPREC)
    for e in elemid
        k2[e] > 0 && (w2 += qs[e]^2 / k2[e])
    end
    # Rank-local under DD: this is a whole-system quantity, so sum the
    # owned partials (an empty rank contributes zero and still rides the
    # collective).
    dom = Cores.domain(atoms)
    dom isa Cores.MPIDomain && (w2 = Cores.allreduce_sum(dom, w2))
    λT = maximum(abs, eigvals(Symmetric(Matrix{Float64}(st.T))))
    return FPPREC(4π) / FPPREC(volume(atoms)) * w2 * FPPREC(λT)
end
