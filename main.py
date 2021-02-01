import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt
from parse_trajectories import get_dihedral_angles
from periodic_kde import evaluate_kde
from reweighting import cs_example, cs_example2

def parse_logfiles():
    if os.path.isfile("pkl/chemical_shifts.pkl"):
        with open("pkl/chemical_shifts.pkl", "rb") as f:
            return pickle.load(f)

    files = glob.glob("log/*.log")
    all_cs = []
    for logfile in files:
        cs = parse_cs(logfile)
        all_cs.append(cs)
    all_cs = np.asarray(all_cs)
    with open("pkl/chemical_shifts.pkl", "wb") as f:
        pickle.dump(all_cs, f)

    return all_cs


def parse_cs(filename):
    cs = []
    with open(filename) as f:
        for line in f:
            if "Isotropic" in line:
                tokens = line.split()
                cs.append(tokens[4])
    return np.asarray(cs, dtype=float)

def get_dihedrals():
    if os.path.isfile("pkl/dihedrals.pkl"):
        with open("pkl/dihedrals.pkl", "rb") as f:
            return pickle.load(f)

    dihedrals = get_dihedral_angles(100)
    with open("pkl/dihedrals.pkl", "wb") as f:
        pickle.dump(dihedrals, f)

    return dihedrals

def plot_kde(dihedrals, filename, vmax=None):
    #TODO plot
    n_bins = 361
    xi, yi = np.mgrid[-180:181:n_bins*1j, -180:181:n_bins*1j]
    zi = evaluate_kde(dihedrals, np.vstack([xi.flatten(), yi.flatten()]).T)
    zi = zi.reshape(xi.shape)
    print(zi.min(), zi.max())
    plt.pcolormesh(xi, yi, zi, shading='gouraud', cmap="gist_earth", 
            vmin=0, vmax=vmax)
    plt.savefig(filename)
    plt.clf()

if __name__ == "__main__":
    np.random.seed(42)
    # Get and process chemical shifts
    cs = parse_logfiles()
    cs_subset = cs[:,[0,1,2,3,4,7,8,9,10,11,12,13,14,17,18,19,20,21]]
    true_mean = cs_subset.mean(0)
    hydrogens = np.asarray([1,0,1,1,0,1,0,1,0,1,1,1,0,1,0,1,1,1], dtype=bool)
    carbons = ~hydrogens

    # Add gaussian noise
    hydrogen_noise = np.random.normal(size=hydrogens.sum())
    carbon_noise = np.random.normal(size=carbons.sum())
    hydrogen_noise *= 0.02/np.std(hydrogen_noise, ddof=0)
    carbon_noise *= 0.1/np.std(carbon_noise, ddof=0)
    true_mean[hydrogens] += hydrogen_noise
    true_mean[carbons] += carbon_noise
    var = np.ones(hydrogens.size) * 0.1**2
    var[hydrogens] = 0.02**2

    # Get dihedrals
    dihedrals = get_dihedrals()

    #TODO draw subset.
    # Weights for drawing subset
    weights = np.ones(dihedrals.shape[0])
    weights[dihedrals[:,0] > 0] *= 3
    weights[dihedrals[:,0] < -100] *= 2
    weights[dihedrals[:,1] > 100] *= 0.5
    weights[dihedrals[:,1] < 110] *= 0.5
    weights /= weights.sum()
    # Indices
    idx = np.arange(dihedrals.shape[0])

    N = 400
    vmax = 0.9e-4
    subset = np.random.choice(idx, size=N, replace=False, p=weights)
    dihedrals_subset = dihedrals[subset]

    sample_cs = cs_subset[subset]

    # Reweight
    reweights = cs_example2(sample_cs, true_mean, var, hydrogens)
    for size in (500,):
        idx1 = np.random.choice(idx, size=size, replace=False)
        plot_kde(dihedrals[idx1], f"true2_{size}.png", vmax=vmax)
        idx2 = np.random.choice(np.arange(N), size=size, replace=True)
        plot_kde(dihedrals_subset[idx2], f"bad2_{size}.png", vmax=vmax)
        idx3 = np.random.choice(np.arange(N), size=size, replace=True, p=reweights)
        plot_kde(dihedrals_subset[idx3], f"reweight2_{size}.png", vmax=vmax)
