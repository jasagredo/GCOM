from __future__ import print_function, division
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

__author__ = "Carlos Armero Canto, Miguel Benito Parejo & Javier Sagredo Tamayo"


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

class PCA(object):

    def __init__(self):
        pass

    def compresion(self, data, tol=0.1):
        mean = np.mean(data, axis=1)

        x_cent = (data.T - mean).T
        u, s, _ = np.linalg.svd(x_cent)
        # S = np.dot(np.dot(u, np.diag(s)),u.T)
        # autovectores son u[:,i], autovalores son s[i]. Ya estan ordenados
        s2 = map(lambda x: x**2, s)
        dp = s.shape[0]
        denom = np.sum(s2)
        val = 0
        for i in range(s.shape[0]-1, -1, -1):
            val += s2[i]
            perdida = val/denom
            if val/denom > tol:
                dp = i
                break
        print(dp)
        w = u[:, 0:dp]
        transformados = np.dot(w.T, data)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(todos[0, :], todos[1, :], todos[2, :], 'o', markersize=8, color='green', alpha=0.2)
        ax.plot([mean[0]], [mean[1]], [mean[2]], 'o', markersize=10, color='red', alpha=0.5)
        for v in u.T:
            a = Arrow3D([mean[0], v[0]], [mean[1], v[1]], [mean[2], v[2]], mutation_scale=20, lw=3, arrowstyle="-|>",
                        color="r")
            ax.add_artist(a)
        ax.set_xlabel('x_values')
        ax.set_ylabel('y_values')
        ax.set_zlabel('z_values')

        plt.title('Eigenvectors')

        plt.show()

        plt.plot(transformados[0, 0:20], transformados[1, 0:20], 'o', markersize=7, color='blue', alpha=0.5,
                 label='class1')
        plt.plot(transformados[0, 20:40], transformados[1, 20:40], '^', markersize=7, color='red', alpha=0.5,
                 label='class2')
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.xlabel('x_values')
        plt.ylabel('y_values')
        plt.legend()
        plt.title('Transformed samples with class labels')

        plt.show()

        from sklearn.decomposition import PCA as sklearnPCA

        sklearn_pca = sklearnPCA(n_components=2)
        sklearn_transf = sklearn_pca.fit_transform(todos.T) * -1

        plt.plot(sklearn_transf[0:20, 0], sklearn_transf[0:20, 1], 'o', markersize=7, color='blue', alpha=0.5,
                 label='class1')
        plt.plot(sklearn_transf[20:40, 0], sklearn_transf[20:40, 1], '^', markersize=7, color='red', alpha=0.5,
                 label='class2')

        plt.xlabel('x_values')
        plt.ylabel('y_values')
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.legend()
        plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

        plt.show()

        return transformados



if __name__ == "__main__":
    np.random.seed(234234782)  # random seed for consistency

    # A reader pointed out that Python 2.7 would raise a
    # "ValueError: object of too small depth for desired array".
    # This can be avoided by choosing a smaller random seed, e.g. 1
    # or by completely omitting this line, since I just used the random seed for
    # consistency.

    mu_vec1 = np.array([0, 0, 0])
    cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
    assert class1_sample.shape == (3, 20), "The matrix has not the dimensions 3x20"

    mu_vec2 = np.array([1, 1, 1])
    cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
    assert class2_sample.shape == (3, 20), "The matrix has not the dimensions 3x20"

    todos = np.concatenate((class1_sample, class2_sample), axis=1)

    a = PCA()

    transformados = a.compresion(todos)
    print(transformados)