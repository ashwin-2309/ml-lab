import numpy as np


class setup:
    def __call__(self):
        pass

    def create_weight(self):
        return np.random.rand(4)


class my_reg(setup):
    def _init_(self, x1, x2, x3, x4, actual):
        super()._init_()
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.actual = actual
        self.l = np.array([self.x1, self.x2, self.x3, self.x4])

    def my_linear(self):
        return np.multiply(self.create_weight(), self.l)

    def square_error(self):
        return (np.sum(np.multiply(self.create_weight(), self.l))-self.actual)**2
