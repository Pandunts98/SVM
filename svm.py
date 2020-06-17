import cvxopt
import numpy as np

from kernel import linear_kernel


class SVM:
    def __init__(self,
                 C,
                 kernel=linear_kernel,
                 kernel_params={}):
        self.C = C and float(C)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.b = None  # intercept
        self.w = None  # you can use this if kernel is linear for faster computation
        self.sv_alphas = None  # solution of qp problem which satisfy condition
        self.sv_s = None  # array of data of support vector points
        self.sv_labels = None  # array of labels of support vector points

    def gram_matrix(self, data):
        """Calculates P matrix.
        WARNING! - You need to implement at least linear_kernel function inside
                   kernel.py before using this function.
        :param data: Data to compute P matrix with [N, M] shape.
        :return: P matrix inside quadratic problem calculated with kernel
                 function.
        """
        n = len(data)
        p = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                p[i, j] = self.kernel(data[i], data[j], **self.kernel_params)
        return p

    def fit(self, data, labels):
        """
        Fit SVM model to data by updating w and b attributes.
        NOTE - There is already a function in cvxopt to solve svm (softmargin)
               problem. But we are recommending to implement using cvxopt
               quadratic programming problem solver.

        :param data: Data with [N, M] shape.
        :param labels: Labels with [N] shape.
        :return: Void.
        """
        p = self.gram_matrix(data)
        n = len(data)
        P = cvxopt.matrix(np.outer(labels, labels) * p)
        q = cvxopt.matrix(-np.ones(n))
        A = cvxopt.matrix(labels, (1, n), 'd')
        b = cvxopt.matrix(0.0)
        if self.C:
            G = cvxopt.matrix(np.vstack((-np.diag(np.ones(n)), np.identity(n))))
            h = cvxopt.matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))
        else:
            G = cvxopt.matrix(-np.diag(np.ones(n)))
            h = cvxopt.matrix(np.zeros(n))
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(sol['x'])
        sv = alpha > 1e-7
        index = np.arange(len(alpha))[sv]
        self.sv_alphas, self.sv_s, self.sv_labels, self.b = alpha[sv], data[sv], labels[sv], 0

        a_len = len(self.sv_alphas)
        for i in range(a_len):
            self.b += self.sv_labels[i]
            self.b -= np.sum(self.sv_alphas * self.sv_labels * p[index[i], sv])
        self.b /= a_len
        if self.kernel == linear_kernel:
            self.w = (self.sv_labels * self.sv_alphas).T @ self.sv_s

    def predict(self, data):
        """
        Predict labels for data.

        :param data: Data with [N, M] shape.
        :return: Predicted labels for given data with [N] shape.
        """
        if self.w is None:
            pred_y = np.zeros(len(data))
            for i in range(len(data)):
                y = 0
                for a, sv_y, sv_x in zip(self.sv_alphas, self.sv_labels, self.sv_s):
                    y += a * sv_y * self.kernel(data[i], sv_x, **self.kernel_params)
                pred_y[i] = y
            return np.sign(pred_y + self.b)
        else:
            return np.sign(data @ self.w + self.b)
