import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Ramachandran

def get_dihedral_angles(step_size=100):
    u0 = mda.Universe("mdshare/alanine-dipeptide-nowater.pdb", "mdshare/alanine-dipeptide-0-250ns-nowater.xtc")
    u1 = mda.Universe("mdshare/alanine-dipeptide-nowater.pdb", "mdshare/alanine-dipeptide-0-250ns-nowater.xtc")
    u2 = mda.Universe("mdshare/alanine-dipeptide-nowater.pdb", "mdshare/alanine-dipeptide-0-250ns-nowater.xtc")
    R0 = Ramachandran(u0.select_atoms('protein')).run(step=step_size)
    R1 = Ramachandran(u1.select_atoms('protein')).run(step=step_size)
    R2 = Ramachandran(u2.select_atoms('protein')).run(step=step_size)
    return np.vstack([R0.angles.squeeze(), R1.angles.squeeze(), R2.angles.squeeze()])

def dump_coordinates():
    u0 = mda.Universe("mdshare/alanine-dipeptide-nowater.pdb",
            "mdshare/alanine-dipeptide-0-250ns-nowater.xtc")
    u1 = mda.Universe("mdshare/alanine-dipeptide-nowater.pdb",
            "mdshare/alanine-dipeptide-0-250ns-nowater.xtc")
    u2 = mda.Universe("mdshare/alanine-dipeptide-nowater.pdb",
            "mdshare/alanine-dipeptide-0-250ns-nowater.xtc")
    c = 0
    for frame in u0.trajectory[::100]:
        write_xyz(c, frame.positions)
        write_com(c, frame.positions)
        c += 1
    for frame in u1.trajectory[::100]:
        write_xyz(c, frame.positions)
        write_com(c, frame.positions)
        c += 1
    for frame in u2.trajectory[::100]:
        write_xyz(c, frame.positions)
        write_com(c, frame.positions)
        c += 1

def write_xyz(idx, coords):
    n = coords.shape[0]
    filename = f"xyz/ala_{idx}.xyz"
    atoms = "HCHHCONHCHCHHHCONHCHHH"
    with open(filename, "w") as f:
        f.write(f"{n}\n\n")
        for i, coord in enumerate(coords):
            x, y, z = coord
            f.write(f"{atoms[i]} {x:.2f} {y:.2f} {z:.2f}\n")

def write_com(idx, coords):
    filename = f"com/ala_{idx}.com"
    atoms = "HCHHCONHCHCHHHCONHCHHH"
    with open(filename, "w") as f:
        f.write("%mem=1GB\n")
        f.write("%nproc=1\n")
        f.write("#t NMR=giao opbe/6-31g(d,p) scrf=(solvent=water)\n\n\n")
        f.write("0 1\n")
        for i, coord in enumerate(coords):
            x, y, z = coord
            f.write(f"{atoms[i]} {x:.2f} {y:.2f} {z:.2f}\n")
        f.write("\n\n")

if __name__ == "__main__":
    dump_coordinates()
