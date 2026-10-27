#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import re, ast, struct
import scienceplots
import matplotlib.pyplot as plt

plt.style.use(["science", "bright", "no-latex"])
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# ============================================================
# (1) landau.dat loader: x, blocks_map
# ============================================================
def load_landau_params(file_path: str):
    with open(file_path, "r") as f:
        header_lines = []
        for line in f:
            if line.startswith("#"):
                header_lines.append(line.strip())
            else:
                break
    header = "\n".join(header_lines)

    m = re.search(r"blocks_map\s*[:=]\s*(\{.*\})", header, flags=re.DOTALL)
    if not m:
        raise ValueError("Header with 'blocks_map={...}' not found.")

    blocks_map = ast.literal_eval(m.group(1))

    data = np.loadtxt(file_path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    idx = data[:, 0].astype(int)
    val = data[:, 1].astype(float)
    order = np.argsort(idx)
    x = val[order]
    return x, blocks_map

def theta_of_T(order: int, T: float, x: np.ndarray, blocks_map: dict) -> np.ndarray:
    i0, ln = blocks_map[order]
    p = x[i0:i0 + ln]

    if order == 2:
        if ln != 3:
            raise ValueError(f"order=2 expects len=3, got {ln}")
        a1 = p[0] + p[1]*T + p[2]*T*T
        return np.array([a1], float)

    if order == 4:
        if ln != 6:
            raise ValueError(f"order=4 expects len=6, got {ln}")
        a11 = p[0] + p[1]*T + p[2]*T*T
        a12 = p[3] + p[4]*T + p[5]*T*T
        return np.array([a11, a12], float)

    if order == 6:
        if ln != 6:
            raise ValueError(f"order=6 expects len=6, got {ln}")
        a111 = p[0] + p[1]*T
        a112 = p[2] + p[3]*T
        a123 = p[4] + p[5]*T
        return np.array([a111, a112, a123], float)

    return p.astype(float)

# ============================================================
# (2) invariants I(P^2) and partial derivatives w.r.t (x=Px^2,y=Py^2,z=Pz^2)
# ============================================================
def inv4_and_dI_dx(x, y, z):
    I1 = x*x + y*y + z*z
    I2 = x*y + y*z + z*x
    dI1 = np.stack([2*x, 2*y, 2*z], axis=-1)
    dI2 = np.stack([y+z, x+z, x+y], axis=-1)
    I = np.stack([I1, I2], axis=-1)
    d = np.stack([dI1, dI2], axis=-2)
    return I, d

def inv6_and_dI_dx(x, y, z):
    I1 = x**3 + y**3 + z**3
    I2 = x**2*(y+z) + y**2*(x+z) + z**2*(x+y)
    I3 = x*y*z
    dI1 = np.stack([3*x**2, 3*y**2, 3*z**2], axis=-1)
    dI2 = np.stack([2*x*(y+z) + (y**2 + z**2),
                    2*y*(x+z) + (x**2 + z**2),
                    2*z*(x+y) + (x**2 + y**2)], axis=-1)
    dI3 = np.stack([y*z, x*z, x*y], axis=-1)
    I = np.stack([I1, I2, I3], axis=-1)
    d = np.stack([dI1, dI2, dI3], axis=-2)
    return I, d

def inv8_and_dI_dx(x, y, z):
    I1 = x**4 + y**4 + z**4
    I2 = x**3*(y+z) + y**3*(x+z) + z**3*(x+y)
    I3 = x**2*y**2 + y**2*z**2 + z**2*x**2
    I4 = x**2*y*z + y**2*x*z + z**2*x*y
    dI1 = np.stack([4*x**3, 4*y**3, 4*z**3], axis=-1)
    dI2 = np.stack([3*x**2*(y+z) + (y**3 + z**3),
                    3*y**2*(x+z) + (x**3 + z**3),
                    3*z**2*(x+y) + (x**3 + y**3)], axis=-1)
    dI3 = np.stack([2*x*(y**2 + z**2),
                    2*y*(x**2 + z**2),
                    2*z*(x**2 + y**2)], axis=-1)
    dI4 = np.stack([2*x*y*z + (y**2*z + z**2*y),
                    2*y*x*z + (x**2*z + z**2*x),
                    2*z*x*y + (x**2*y + y**2*x)], axis=-1)
    I = np.stack([I1, I2, I3, I4], axis=-1)
    d = np.stack([dI1, dI2, dI3, dI4], axis=-2)
    return I, d

def inv10_and_dI_dx(x, y, z):
    I1 = x**5 + y**5 + z**5
    I2 = x**4*(y+z) + y**4*(x+z) + z**4*(x+y)
    I3 = x**3*(y**2+z**2) + y**3*(x**2+z**2) + z**3*(x**2+y**2)
    I4 = x**3*y*z + y**3*x*z + z**3*x*y
    I5 = x**2*y**2*z + y**2*z**2*x + z**2*x**2*y
    dI1 = np.stack([5*x**4, 5*y**4, 5*z**4], axis=-1)
    dI2 = np.stack([4*x**3*(y+z) + (y**4 + z**4),
                    4*y**3*(x+z) + (x**4 + z**4),
                    4*z**3*(x+y) + (x**4 + y**4)], axis=-1)
    dI3 = np.stack([
        3*x**2*(y**2+z**2) + 2*x*(y**3+z**3),
        3*y**2*(x**2+z**2) + 2*y*(x**3+z**3),
        3*z**2*(x**2+y**2) + 2*z*(x**3+y**3)
    ], axis=-1)
    dI4 = np.stack([3*x**2*y*z + (y**3*z + z**3*y),
                    3*y**2*x*z + (x**3*z + z**3*x),
                    3*z**2*x*y + (x**3*y + y**3*x)], axis=-1)
    dI5 = np.stack([
        2*x*y**2*z + y**2*z**2 + 2*x*z**2*y,
        2*y*x**2*z + 2*y*z**2*x + z**2*x**2,
        x**2*y**2 + 2*z*y**2*x + 2*z*x**2*y
    ], axis=-1)
    I = np.stack([I1, I2, I3, I4, I5], axis=-1)
    d = np.stack([dI1, dI2, dI3, dI4, dI5], axis=-2)
    return I, d

def inv12_and_dI_dx(x, y, z):
    I1 = x**6 + y**6 + z**6
    I2 = x**5*(y+z) + y**5*(x+z) + z**5*(x+y)
    I3 = x**4*(y**2+z**2) + y**4*(x**2+z**2) + z**4*(x**2+y**2)
    I4 = x**3*y**3 + y**3*z**3 + z**3*x**3
    I5 = x**4*y*z + y**4*x*z + z**4*x*y
    I6 = x**3*(y**2*z + z**2*y) + y**3*(x**2*z + z**2*x) + z**3*(x**2*y + y**2*x)
    I7 = x**2*y**2*z**2
    dI1 = np.stack([6*x**5, 6*y**5, 6*z**5], axis=-1)
    dI2 = np.stack([5*x**4*(y+z) + (y**5+z**5),
                    5*y**4*(x+z) + (x**5+z**5),
                    5*z**4*(x+y) + (x**5+y**5)], axis=-1)
    dI3 = np.stack([
        4*x**3*(y**2+z**2) + 2*x*(y**4+z**4),
        4*y**3*(x**2+z**2) + 2*y*(x**4+z**4),
        4*z**3*(x**2+y**2) + 2*z*(x**4+y**4)
    ], axis=-1)
    dI4 = np.stack([3*x**2*(y**3+z**3),
                    3*y**2*(x**3+z**3),
                    3*z**2*(x**3+y**3)], axis=-1)
    dI5 = np.stack([4*x**3*y*z + (y**4*z + z**4*y),
                    4*y**3*x*z + (x**4*z + z**4*x),
                    4*z**3*x*y + (x**4*y + y**4*x)], axis=-1)
    dI6 = np.stack([
        3*x**2*(y**2*z + z**2*y) + y**3*(2*x*z + z**2) + z**3*(2*x*y + y**2),
        3*y**2*(x**2*z + z**2*x) + x**3*(2*y*z + z**2) + z**3*(x**2 + 2*y*x),
        3*z**2*(x**2*y + y**2*x) + x**3*(y**2 + 2*z*y) + y**3*(x**2 + 2*z*x)
    ], axis=-1)
    dI7 = np.stack([2*x*y**2*z**2, 2*y*x**2*z**2, 2*z*x**2*y**2], axis=-1)
    I = np.stack([I1,I2,I3,I4,I5,I6,I7], axis=-1)
    d = np.stack([dI1,dI2,dI3,dI4,dI5,dI6,dI7], axis=-2)
    return I, d

def inv14_and_dI_dx(x, y, z):
    x2,y2,z2 = x*x, y*y, z*z
    x3,y3,z3 = x2*x, y2*y, z2*z
    x4,y4,z4 = x2*x2, y2*y2, z2*z2
    x5,y5,z5 = x4*x, y4*y, z4*z
    x6,y6,z6 = x3*x3, y3*y3, z3*z3
    x7,y7,z7 = x6*x, y6*y, z6*z

    I1 = x7 + y7 + z7
    I2 = x6*(y+z) + y6*(x+z) + z6*(x+y)
    I3 = x5*(y2+z2) + y5*(x2+z2) + z5*(x2+y2)
    I4 = x4*(y3+z3) + y4*(x3+z3) + z4*(x3+y3)
    I5 = x5*y*z + y5*x*z + z5*x*y
    I6 = x4*(y2*z + z2*y) + y4*(x2*z + z2*x) + z4*(x2*y + y2*x)
    I7 = x3*y2*z2 + y3*x2*z2 + z3*x2*y2
    I8 = x3*y3*z + y3*z3*x + z3*x3*y

    dI1 = np.stack([7*x6, 7*y6, 7*z6], axis=-1)
    dI2 = np.stack([6*x5*(y+z) + (y6+z6),
                    6*y5*(x+z) + (x6+z6),
                    6*z5*(x+y) + (x6+y6)], axis=-1)
    dI3 = np.stack([5*x4*(y2+z2) + 2*x*(y5+z5),
                    5*y4*(x2+z2) + 2*y*(x5+z5),
                    5*z4*(x2+y2) + 2*z*(x5+y5)], axis=-1)
    dI4 = np.stack([4*x3*(y3+z3) + 3*x2*(y4+z4),
                    4*y3*(x3+z3) + 3*y2*(x4+z4),
                    4*z3*(x3+y3) + 3*z2*(x4+y4)], axis=-1)
    dI5 = np.stack([5*x4*y*z + (y5*z + z5*y),
                    5*y4*x*z + (x5*z + z5*x),
                    5*z4*x*y + (x5*y + y5*x)], axis=-1)
    dI6 = np.stack([
        4*x3*(y2*z + z2*y) + y4*(2*x*z + z2) + z4*(2*x*y + y2),
        4*y3*(x2*z + z2*x) + x4*(2*y*z + z2) + z4*(x2 + 2*y*x),
        4*z3*(x2*y + y2*x) + x4*(y2 + 2*z*y) + y4*(x2 + 2*z*x)
    ], axis=-1)
    dI7 = np.stack([
        3*x2*y2*z2 + y3*(2*x*z2) + z3*(2*x*y2),
        3*y2*x2*z2 + x3*(2*y*z2) + z3*(2*y*x2),
        3*z2*x2*y2 + x3*(2*z*y2) + y3*(2*z*x2)
    ], axis=-1)
    dI8 = np.stack([
        3*x2*y3*z + y3*z3 + 3*z3*x2*y,
        3*y2*z3*x + z3*x3 + 3*x3*y2*z,
        3*z2*x3*y + x3*y3 + 3*y3*z2*x
    ], axis=-1)

    I = np.stack([I1,I2,I3,I4,I5,I6,I7,I8], axis=-1)
    d = np.stack([dI1,dI2,dI3,dI4,dI5,dI6,dI7,dI8], axis=-2)
    return I, d

INV_AND_D = {
    4: inv4_and_dI_dx,
    6: inv6_and_dI_dx,
    8: inv8_and_dI_dx,
    10: inv10_and_dI_dx,
    12: inv12_and_dI_dx,
    14: inv14_and_dI_dx,
}

# ============================================================
# (3) δFloc/δP using ALL orders in landau.dat
# ============================================================
def dF_local_dP(P, dx, T, x_landau, blocks_map):
    Px,Py,Pz = P[...,0], P[...,1], P[...,2]
    x = Px*Px; y = Py*Py; z = Pz*Pz
    dV = dx**3

    dfdPx = np.zeros_like(Px)
    dfdPy = np.zeros_like(Py)
    dfdPz = np.zeros_like(Pz)

    if 2 in blocks_map:
        a1 = theta_of_T(2, T, x_landau, blocks_map)[0]
        dfdPx += a1 * (2*Px)
        dfdPy += a1 * (2*Py)
        dfdPz += a1 * (2*Pz)

    for order in sorted(blocks_map.keys()):
        if order == 2:
            continue
        if order not in INV_AND_D:
            raise ValueError(f"Missing invariant derivative for order={order}")
        theta = theta_of_T(order, T, x_landau, blocks_map)
        _, dIdxyz = INV_AND_D[order](x, y, z)
        dIdx = dIdxyz[...,0]
        dIdy = dIdxyz[...,1]
        dIdz = dIdxyz[...,2]
        cdx = np.tensordot(dIdx, theta, axes=([-1],[0]))
        cdy = np.tensordot(dIdy, theta, axes=([-1],[0]))
        cdz = np.tensordot(dIdz, theta, axes=([-1],[0]))
        dfdPx += cdx * (2*Px)
        dfdPy += cdy * (2*Py)
        dfdPz += cdz * (2*Pz)

    return np.stack([dfdPx, dfdPy, dfdPz], axis=-1) * dV

# ============================================================
# (4) δFgrad/δP : 1nn + 2nn, linear only
# ============================================================
def _add_axis_bonds_contrib(Gx, Gy, Gz, Px, Py, Pz, axis, shift, pref, J1, J2, J3, J4):
    Px_p = np.roll(Px, -shift, axis=axis)
    Py_p = np.roll(Py, -shift, axis=axis)
    Pz_p = np.roll(Pz, -shift, axis=axis)

    dpx = Px_p - Px
    dpy = Py_p - Py
    dpz = Pz_p - Pz

    if axis == 0:
        gx_d = pref * (2*J1*dpx + J3*(dpy+dpz))
        gy_d = pref * (2*J2*dpy + J3*dpx + J4*dpz)
        gz_d = pref * (2*J2*dpz + J3*dpx + J4*dpy)
    elif axis == 1:
        gy_d = pref * (2*J1*dpy + J3*(dpx+dpz))
        gx_d = pref * (2*J2*dpx + J3*dpy + J4*dpz)
        gz_d = pref * (2*J2*dpz + J3*dpy + J4*dpx)
    else:
        gz_d = pref * (2*J1*dpz + J3*(dpx+dpy))
        gx_d = pref * (2*J2*dpx + J3*dpz + J4*dpy)
        gy_d = pref * (2*J2*dpy + J3*dpz + J4*dpx)

    Gx += -gx_d + np.roll(gx_d, shift, axis=axis)
    Gy += -gy_d + np.roll(gy_d, shift, axis=axis)
    Gz += -gz_d + np.roll(gz_d, shift, axis=axis)

def dF_grad_dP_1nn_2nn_linear(P, dx, J, pref=None):
    if pref is None:
        pref = dx

    J = np.asarray(J, dtype=float)
    if J.shape[0] != 8:
        raise ValueError("J must be length 8: [J1,J2,J3,J4, J1_2,J2_2,J3_2,J4_2]")

    J1,J2,J3,J4, J1_2,J2_2,J3_2,J4_2 = J

    Px,Py,Pz = P[...,0], P[...,1], P[...,2]
    Gx = np.zeros_like(Px)
    Gy = np.zeros_like(Py)
    Gz = np.zeros_like(Pz)

    for ax in (0,1,2):
        _add_axis_bonds_contrib(Gx, Gy, Gz, Px, Py, Pz, axis=ax, shift=1, pref=pref, J1=J1,   J2=J2,   J3=J3,   J4=J4)
        _add_axis_bonds_contrib(Gx, Gy, Gz, Px, Py, Pz, axis=ax, shift=2, pref=pref, J1=J1_2, J2=J2_2, J3=J3_2, J4=J4_2)

    return np.stack([Gx,Gy,Gz], axis=-1)

# ============================================================
# (5) electrostatics
# ============================================================
def depol_field_fft(P, dx, EPS0, epsb=1.0):
    Nx,Ny,Nz,_ = P.shape
    Px,Py,Pz = P[...,0], P[...,1], P[...,2]
    Phx = np.fft.fftn(Px); Phy = np.fft.fftn(Py); Phz = np.fft.fftn(Pz)

    kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=dx)
    kz = 2*np.pi*np.fft.fftfreq(Nz, d=dx)
    kx3 = kx[:,None,None]; ky3 = ky[None,:,None]; kz3 = kz[None,None,:]
    k2 = kx3*kx3 + ky3*ky3 + kz3*kz3
    kdotP = kx3*Phx + ky3*Phy + kz3*Phz

    phi = np.zeros_like(kdotP, dtype=np.complex128)
    mask = (k2 != 0.0)
    phi[mask] = (-1j) * kdotP[mask] / (EPS0*epsb*k2[mask])

    Exh = (-1j)*kx3*phi
    Eyh = (-1j)*ky3*phi
    Ezh = (-1j)*kz3*phi

    Ex = np.fft.ifftn(Exh).real
    Ey = np.fft.ifftn(Eyh).real
    Ez = np.fft.ifftn(Ezh).real
    return np.stack([Ex,Ey,Ez], axis=-1)

def dF_elec_dP(P, dx, EPS0, epsb=1.0):
    dV = dx**3
    E = depol_field_fft(P, dx, EPS0, epsb)
    return (-E) * dV

# ============================================================
# (6) Initial conditions
# ============================================================
def init_180_dw(Nx,Ny,Nz, P0, w, axis=0, comp=2):
    P = np.zeros((Nx,Ny,Nz,3), float)
    L = [Nx,Ny,Nz][axis]
    coord = np.arange(L, dtype=float)
    x1 = L/4.0
    x2 = 3.0*L/4.0
    prof = (np.tanh((coord-x1)/w) - np.tanh((coord-x2)/w) - 1.0)
    prof *= P0

    if axis == 0:
        P[...,comp] = prof[:,None,None]
    elif axis == 1:
        P[...,comp] = prof[None,:,None]
    else:
        P[...,comp] = prof[None,None,:]
    return P

def init_uniform(Nx,Ny,Nz, P0, comp=2):
    P = np.zeros((Nx,Ny,Nz,3), float)
    P[..., comp] = P0
    return P

# ============================================================
# (8) VTI writer: VTK XML (binary appended raw)
# ============================================================
def write_vti_binary_appended_raw(filename, P, dx, origin=(0.0, 0.0, 0.0), name="P"):
    P = np.asarray(P)
    if P.ndim != 4 or P.shape[-1] != 3:
        raise ValueError("P must have shape (Nx,Ny,Nz,3)")

    Nx, Ny, Nz, _ = P.shape
    ox, oy, oz = origin
    extent = f"0 {Nx-1} 0 {Ny-1} 0 {Nz-1}"

    P32 = P.astype(np.float32, copy=False)
    vtk_order = np.transpose(P32, (2, 1, 0, 3)).ravel(order="C")
    payload = vtk_order.tobytes(order="C")
    nbytes = len(payload)

    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" header_type="UInt32">\n'
        f'  <ImageData WholeExtent="{extent}" Origin="{ox} {oy} {oz}" Spacing="{dx} {dx} {dx}">\n'
        f'    <Piece Extent="{extent}">\n'
        '      <PointData>\n'
        f'        <DataArray type="Float32" Name="{name}" NumberOfComponents="3" format="appended" offset="0"/>\n'
        '      </PointData>\n'
        '      <CellData/>\n'
        '    </Piece>\n'
        '  </ImageData>\n'
        '  <AppendedData encoding="raw">\n'
        '    _'
    ).encode("utf-8")

    tail = (
        '\n  </AppendedData>\n'
        '</VTKFile>\n'
    ).encode("utf-8")

    with open(filename, "wb") as f:
        f.write(xml)
        f.write(struct.pack("<I", nbytes))
        f.write(payload)
        f.write(tail)

# ============================================================
# (9) Energy evaluation: Floc + Fgrad + Felec
# ============================================================
def F_local(P, dx, T, x_landau, blocks_map):
    Px,Py,Pz = P[...,0], P[...,1], P[...,2]
    x = Px*Px; y = Py*Py; z = Pz*Pz
    dV = dx**3

    f = np.zeros_like(Px, dtype=float)

    if 2 in blocks_map:
        a1 = theta_of_T(2, T, x_landau, blocks_map)[0]
        f += a1 * (x + y + z)

    for order in sorted(blocks_map.keys()):
        if order == 2:
            continue
        if order not in INV_AND_D:
            raise ValueError(f"Missing invariant for order={order}")
        theta = theta_of_T(order, T, x_landau, blocks_map)  # (ninv,)
        I, _ = INV_AND_D[order](x, y, z)                    # (...,ninv)
        f += np.tensordot(I, theta, axes=([-1],[0]))

    return float(np.sum(f) * dV)

def _Fgrad_axis_shift(P, axis, shift, pref, J1,J2,J3,J4):
    Px,Py,Pz = P[...,0], P[...,1], P[...,2]
    Px_p = np.roll(Px, -shift, axis=axis)
    Py_p = np.roll(Py, -shift, axis=axis)
    Pz_p = np.roll(Pz, -shift, axis=axis)

    dpx = Px_p - Px
    dpy = Py_p - Py
    dpz = Pz_p - Pz

    if axis == 0:      # parallel=Px, perps=Py,Pz
        dpar, da, db = dpx, dpy, dpz
    elif axis == 1:    # parallel=Py, perps=Px,Pz
        dpar, da, db = dpy, dpx, dpz
    else:              # parallel=Pz, perps=Px,Py
        dpar, da, db = dpz, dpx, dpy

    e = (J1*(dpar*dpar) +
         J2*(da*da + db*db) +
         J3*(dpar*da + dpar*db) +
         J4*(da*db))
    return float(pref * np.sum(e))

def F_grad_1nn_2nn_linear(P, dx, J, pref=None):
    if pref is None:
        pref = dx
    J = np.asarray(J, float)
    J1,J2,J3,J4, J1_2,J2_2,J3_2,J4_2 = J
    F = 0.0
    for ax in (0,1,2):
        F += _Fgrad_axis_shift(P, ax, 1, pref, J1,   J2,   J3,   J4)
        F += _Fgrad_axis_shift(P, ax, 2, pref, J1_2, J2_2, J3_2, J4_2)
    return float(F)

def F_elec(P, dx, EPS0, epsb=1.0):
    # positive-definite form
    dV = dx**3
    E = depol_field_fft(P, dx, EPS0, epsb)
    return float(0.5 * EPS0 * epsb * np.sum(E[...,0]**2 + E[...,1]**2 + E[...,2]**2) * dV)

def F_total(P, dx, T, x_landau, blocks_map, J, EPS0, epsb=1.0, pref=None):
    Fl = F_local(P, dx, T, x_landau, blocks_map)
    Fg = F_grad_1nn_2nn_linear(P, dx, J, pref=pref)
    Fe = F_elec(P, dx, EPS0, epsb=epsb)
    return Fl + Fg + Fe, (Fl, Fg, Fe)

# ============================================================
# (7) TDGL integrator (final step -> VTI)
# ============================================================
def tdgl_relax(
    landau_dat,
    T,
    J,  # length 8
    Nx=128, Ny=1, Nz=1,
    dx=4.0,
    mobility=1.0,
    dt=1e-4,
    nsteps=20000,
    P_init=None,          # ndarray (Nx,Ny,Nz,3)
    epsb=1.0,
    FOURPI_EPS0_INV=14.399645478,
    save_every=1000,
    out_vti=None,         # filename or None
):
    x_landau, blocks_map = load_landau_params(landau_dat)
    EPS0 = 1.0 / (4.0*np.pi*FOURPI_EPS0_INV)

    if P_init is None:
        raise ValueError("P_init must be provided")

    P = np.array(P_init, float, copy=True)
    hist = {"step": [], "max_dP": []}

    for step in range(nsteps):
        G_loc  = dF_local_dP(P, dx, T, x_landau, blocks_map)
        G_grad = dF_grad_dP_1nn_2nn_linear(P, dx, J, pref=dx)
        G_elec = dF_elec_dP(P, dx, EPS0=EPS0, epsb=epsb)

        dP = -mobility * (G_loc + G_grad + G_elec)
        P_new = P + dt * dP

        maxchg = float(np.max(np.abs(P_new - P)))
        P = P_new

        if (step % save_every) == 0:
            hist["step"].append(step)
            hist["max_dP"].append(maxchg)
            print(f"step={step:7d}  max|ΔP|={maxchg:.3e}")

        if maxchg < 1e-10:
            print(f"Converged at step={step}, max|ΔP|={maxchg:.3e}")
            break

    if out_vti is not None:
        write_vti_binary_appended_raw(out_vti, P, dx=dx, origin=(0.0,0.0,0.0), name="P")
        print(f"Wrote: {out_vti}")

    return P, hist, (x_landau, blocks_map, EPS0)

# ============================================================
# Wall energy helper
# ============================================================
def wall_area(Nx,Ny,Nz, dx, axis):
    # wall normal = axis, so area is product of the other two dimensions * dx^2
    dims = [Nx,Ny,Nz]
    a = 1.0
    for ax in (0,1,2):
        if ax != axis:
            a *= dims[ax] * dx
    return float(a)

# ============================================================
# Example
# ============================================================
if __name__ == "__main__":
    # J = [J1,J2,J3,J4,  J1_2,J2_2,J3_2,J4_2]
    J_fit = np.array([40.0, 40.0, 0.0, 0.0,   0.0, 0.0, 0.0, 0.0], dtype=float)  # example ONLY

    # ---- settings ----
    landau_dat = "landau.dat"
    T = 300.0
    Nx,Ny,Nz = 100, 1, 1
    dx = 16.09434994
    mobility = 1.0
    dt = 1e-4
    nsteps = 10000
    epsb = 1.0

    axis = 0
    comp = 2
    P0 = 0.02
    w0 = 4.0

    # ---- DW run ----
    P_dw0 = init_180_dw(Nx,Ny,Nz, P0=P0, w=w0, axis=axis, comp=comp)
    P_dw0[..., comp] += 1e-2*(np.random.rand(Nx,Ny,Nz)-0.5)

    P_dw, hist_dw, (x_landau, blocks_map, EPS0) = tdgl_relax(
        landau_dat=landau_dat, T=T, J=J_fit,
        Nx=Nx, Ny=Ny, Nz=Nz, dx=dx,
        mobility=mobility, dt=dt, nsteps=nsteps,
        P_init=P_dw0, epsb=epsb,
        save_every=2000,
        out_vti="P_final_dw.vti",
    )

    # ---- Uniform run (same size) ----
    P_uni0 = init_uniform(Nx,Ny,Nz, P0=P0, comp=comp)
    P_uni0[..., comp] += 1e-3*(np.random.rand(Nx,Ny,Nz)-0.5)  # tiny noise (optional)

    P_uni, hist_uni, _ = tdgl_relax(
        landau_dat=landau_dat, T=T, J=J_fit,
        Nx=Nx, Ny=Ny, Nz=Nz, dx=dx,
        mobility=mobility, dt=dt, nsteps=nsteps,
        P_init=P_uni0, epsb=epsb,
        save_every=2000,
        out_vti="P_final_uniform.vti",
    )

    # ---- Energy + wall energy ----
    Fdw, (Fl_dw, Fg_dw, Fe_dw) = F_total(P_dw, dx, T, x_landau, blocks_map, J_fit, EPS0, epsb=epsb, pref=dx)
    Fun, (Fl_un, Fg_un, Fe_un) = F_total(P_uni, dx, T, x_landau, blocks_map, J_fit, EPS0, epsb=epsb, pref=dx)

    A = wall_area(Nx,Ny,Nz, dx, axis=axis)
    Nwall = 2.0  # periodic 180° profile => two walls in the cell
    gamma = (Fdw - Fun) / (Nwall * A)

    print("\n=== Energies (eV) ===")
    print(f"DW   : F={Fdw:.6e}  (Floc={Fl_dw:.6e}, Fgrad={Fg_dw:.6e}, Felec={Fe_dw:.6e})")
    print(f"Uni  : F={Fun:.6e}  (Floc={Fl_un:.6e}, Fgrad={Fg_un:.6e}, Felec={Fe_un:.6e})")
    print("\n=== Wall energy ===")
    print(f"Area A = {A:.6e} Å^2   (axis={axis})")
    print(f"gamma = (F_DW - F_Uni) / ({int(Nwall)} * A) = {gamma:.6e} eV/Å^2")

    print("\ndone. shapes:", P_dw.shape, P_uni.shape)
