import numpy as np
import matplotlib.pyplot as plt

class Grid:
    def __init__(self, r=1.0, n=7, c=(0.0, 0.0, 0.0)):
        t = np.linspace(-0.5 * np.sqrt(2.0 * np.pi / 3.0),
                         0.5 * np.sqrt(2.0 * np.pi / 3.0), n)
        x, y = np.meshgrid(t, t)
        with np.errstate(all='ignore'):
            sqrt2 = np.sqrt(2.0)
            beta = np.sqrt(np.pi / 6.0)
            alpha = y * np.pi / (12.0 * x)
            coeff = (np.power(2.0, 1.0/4.0) * x / beta)
            denom = np.sqrt(sqrt2 - np.cos(alpha))
            X1 = coeff * (sqrt2 * np.cos(alpha) - 1.0) / denom
            Y1 = coeff * (sqrt2 * np.sin(alpha)) / denom
            alpha = x * np.pi / (12.0 * y)
            coeff = (np.power(2.0, 1.0/4.0) * y / beta)
            denom = np.sqrt(sqrt2 - np.cos(alpha))
            X2 = coeff * (sqrt2 * np.sin(alpha)) / denom
            Y2 = coeff * (sqrt2 * np.cos(alpha) - 1.0) / denom
            X = np.nan * np.ones_like(X1)
            Y = np.nan * np.ones_like(Y1)
            xgty = np.where(np.logical_and(y >= 0, np.abs(y) <= np.abs(x)))
            nxgty = np.where(np.logical_and(y < 0, np.abs(y) <= np.abs(x)))
            ygtx = np.where(np.logical_and(y >= 0, np.abs(x) <= np.abs(y)))
            nygtx = np.where(np.logical_and(y < 0, np.abs(x) <= np.abs(y)))
            X[xgty] = X1[xgty]
            Y[xgty] = Y1[xgty]
            X[nxgty] = X1[nxgty]
            Y[nxgty] = Y1[nxgty]
            X[ygtx] = X2[ygtx]
            Y[ygtx] = Y2[ygtx]
            X[nygtx] = X2[nygtx]
            Y[nygtx] = Y2[nygtx]

        chi = np.sqrt(1.0 - (np.power(X, 2.0) + np.power(Y, 2.0))/4.0) * X
        upsilon = np.sqrt(1.0 - (np.power(X, 2.0) + np.power(Y, 2.0))/4.0) * Y
        zeta = 1.0 - (np.power(X, 2.0) + np.power(Y, 2.0))/2.0
        # we have identity, neg Z, switch y and z, switch x and z
        xL = np.concatenate((chi, chi, chi, chi, zeta, -zeta))
        yL = np.concatenate((upsilon, upsilon, zeta, -zeta, upsilon, upsilon))
        zL = np.concatenate((zeta, -zeta, upsilon, upsilon, chi, chi))
        # the seams are repeated, so we want to take all unique triples:
        pts = np.column_stack((xL.flatten(), yL.flatten(), zL.flatten()))
        # screen out anything nonfinite because of div by zero
        idx = np.all(np.isfinite(pts), axis=1)
        pts = pts[idx]
        # if n is odd, we are missing the points at -1, 0, 0, etc.
        if n % 2 != 0:
            midps = np.array([[ 1.0,  0.0,  0.0],
                            [-1.0,  0.0,  0.0],
                            [ 0.0,  1.0,  0.0],
                            [ 0.0,  -1.0,  0.0],
                            [ 0.0,  0.0,  1.0],
                            [ 0.0,  0.0,  -1.0]])
            pts = np.row_stack((pts, midps))
        pts = r * np.unique(pts, axis=0) + np.array(c)
        xL = pts[:, 0]
        yL = pts[:, 1]
        zL = pts[:, 2]
        self.len = pts.shape[0]
        self.x = xL
        self.y = yL
        self.z = zL

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx >= self.len:
            raise StopIteration
        else:
            return self.x[idx], self.y[idx], self.z[idx]

    def plot_triple(self, xL=None, yL=None, zL=None):
        if xL is None:
            xL = self.x
            yL = self.y
            zL = self.z
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(221, projection='3d')
        ax.scatter(xL, yL, zL, alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=90.0, azim=0.0)
        ax = fig.add_subplot(222, projection='3d')
        ax.scatter(xL, yL, zL, alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=0.0, azim=90.0)
        ax = fig.add_subplot(223, projection='3d')
        ax.scatter(xL, yL, zL, alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=90.0, azim=90.0)
        ax = fig.add_subplot(224, projection='3d')
        ax.scatter(xL, yL, zL, alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()