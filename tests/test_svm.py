from unittest import TestCase

import numpy as np

from svm import SVM


class SCMTest(TestCase):
    def test_gram_matrix(self):
        s = SVM(C=0)
        np.testing.assert_allclose(
            s.gram_matrix(np.array([[4, 5], [1, 2], [2, 3]])),
            np.array([[41., 14., 23.],
                      [14., 5., 8.],
                      [23., 8., 13.]]))

    def test_fit(self):
        s = SVM(C=None)
        s.fit(np.array([[1, 2], [1, 1], [2, 1.5],
                        [7, 8], [8, 6.5], [8, 7.5], [9, 7]]),
              np.array([-1, -1, -1, 1, 1, 1, 1]))
        np.testing.assert_allclose(s.w, np.array([0.1967213, 0.16393444]), rtol=1e-5)
        np.testing.assert_allclose(s.b, -1.6393436694251626, rtol=1e-5)

    def test_predict(self):
        s = SVM(C=None)
        s.fit(np.array([[1, 2], [1, 1], [2, 1.5],
                        [7, 8], [8, 6.5], [8, 7.5], [9, 7]]),
              np.array([-1, -1, -1, 1, 1, 1, 1]))
        np.testing.assert_equal(s.predict(np.array([[1, 1.1]])), -1)
