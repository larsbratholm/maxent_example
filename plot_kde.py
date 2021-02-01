import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from parse_trajectories import get_dihedral_angles
from periodic_kde import evaluate_kde

if __name__ == "__main__":
    if os.path.isfile("dihedrals.pkl"):
        with open("dihedrals.pkl", "rb") as f:
            xi, yi, zi = pickle.load(f)
    else:
        n_bins = 361
        dihedrals = get_dihedral_angles(100)
        #grid = np.mgrid[-180:181:1.0, -180:181:1.0].reshape(2,-1).T
        xi, yi = np.mgrid[-180:181:n_bins*1j, -180:181:n_bins*1j]
        zi = evaluate_kde(dihedrals, np.vstack([xi.flatten(), yi.flatten()]).T)
        with open("dihedrals.pkl", "wb") as f:
            pickle.dump((xi, yi, zi), f)
    zi = zi.reshape(xi.shape)
    zi[xi > 0] *= 4
    zi[xi < -100] *= 2
    zi[yi > 100] *= 0.5
    plt.pcolormesh(xi, yi, zi, shading='gouraud', cmap="gist_earth")
    plt.show()
    #for i in [5, 10, 15, 20, 30, 40, 60, 80, 100]:
    #    #plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=cm)
    #    plt.contour(xi, yi, zi.reshape(xi.shape), levels=i)
    #    plt.savefig(f"contour_{i}.png")
    #    plt.clf()

