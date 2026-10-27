# Regenerates example/Relax/input.xyz from the ideal cubic cell.
#
#     julia --project make_input.jl        (run from the repository root)
#
# The shipped input is deliberately NOT the ideal structure: an ideal cubic
# perovskite is a stationary point by symmetry, so relaxing it moves nothing
# and the example would demonstrate nothing. (Worse, for a ferroelectric it
# is a SADDLE, not a minimum — see the note at the end of input.jl.)
using FMD, FMD.Cores
using Random, Printf
# Rattle the ideal cubic cell so the example demonstrates a relaxation
# rather than stopping at step 0. Fixed seed: the shipped file must be
# reproducible from this script.
atoms = readxyz(open("example/NPT-MD/input.xyz", "r"))[1]
rng = Random.Xoshiro(20250819)
lat = Array(atoms.info[:lattice])
amp = 0.05 / UNITS[:Å]                      # 0.05 Å rms, in Bohr
dfrac = (amp .* randn(rng, FMD.FPPREC, size(atoms.arrays[:pos]))) * inv(lat)
atoms.arrays[:pos] .+= dfrac
# Shells follow their cores (the file carries absolute shell positions).
elements = Dict("K" => 1, "Nb" => 2, "O" => 3)
open("example/Relax/input.xyz", "w") do io
    writexyz(io, atoms, (; elements = elements))
end
@printf("wrote %d atoms, max |dpos| = %.4f Å\n", length(atoms),
        maximum(abs, dfrac * lat) * UNITS[:Å])
