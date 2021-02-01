import numpy as np
from ligo.skymap.kde import BoundedKDE

def evaluate_kde(points, grid):
    m = BoundedKDE(points.T, low=-180, high=180, periodic=True)
    return m.evaluate(grid.T)

if __name__ == "__main__":
    x = (np.random.rand(1000,2)-0.5)*360

    m = BoundedKDE(x.T, low=-180, high=180, periodic=True)
    y = m.evaluate(x[:2].T)
    print(y)
