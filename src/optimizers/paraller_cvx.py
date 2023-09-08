from scipy import optimize
import multiprocessing
import numpy as np
import scipy
import math
import time


class CVXOptimizer:
    def __init__(self, trainer):
        self.num_clients = trainer.num_clients
        self.A = [0] * trainer.num_clients
        self.B = [0] * trainer.num_clients

    def objective(self, q):
        res = 0
        return res

    def constraint(self, q):
        return sum(q) - 1

    def constraint_ineq(self, q):
        return q

    def CVXMinimize(self):
        problem = {
            'fun': self.objective,
            'constraints': [{'type': 'eq', 'fun': self.constraint},
                            {'type': 'ineq', 'fun': self.constraint_ineq}],
            'x0': [1 / self.num_clients] * self.num_clients,
            'method': 'trust-constr',
            # 'options': {'maxiter': 2000, 'xtol': 1e-6}
        }
        soln = scipy.optimize.minimize(**problem)
        print(soln)
        return soln


class GradientPerturbedOptimizor(CVXOptimizer):
    def __init__(self, trainer):
        self.l = trainer.balance_l
        super(GradientPerturbedOptimizor, self).__init__(trainer)
        d = trainer.d
        p = trainer.p
        s = trainer.s
        K = trainer.clients_per_round
        E = trainer.num_epoch
        R = trainer.num_round
        N = trainer.num_clients
        delta = trainer.delta
        delta1 = delta / (2 * R * K)
        delta2 = delta3 = delta / 4
        epsilon = trainer.epsilon

        for i in range(N):
            self.A[i] = p[i] ** 2 / K
            self.B[i] = math.log(1 / delta1) * math.log(1 / delta2) * math.log(1 / delta3) * E * (R / K) ** 0.5 * \
                        s[i] ** 2 * p[i] ** 2 / epsilon[i] ** 2

    def objective(self, q):
        res = 0
        for i in range(self.num_clients):
            if q[i] < 0:  # q may be smaller than zero due to precision errors of CVX
                return 1e8  # return inf will cause crash, so return a big enough number
            res += self.A[i] / q[i] + self.l * self.B[i] / q[i] ** 0.5

        return res

    def get_q(self):
        q = self.CVXMinimize().x
        print(q)
        return q


class OutputPerturbedOptimizor(CVXOptimizer):
    def __init__(self, trainer):
        self.l = 80
        super(OutputPerturbedOptimizor, self).__init__(trainer)
        d = trainer.d
        p = trainer.p
        s = trainer.s
        gamma = trainer.gamma
        K = trainer.clients_per_round
        E = trainer.num_epoch
        R = trainer.num_round
        N = trainer.num_clients
        # G = trainer.G
        delta = trainer.delta
        delta1 = delta3 = delta / 2
        epsilon = trainer.epsilon

        for i in range(N):
            self.A[i] = gamma * p[i] ** 2 / K
            self.B[i] = math.log(1 / delta1) * math.log(1 / delta3) * (R / K) ** 0.5 * \
                        s[i] ** 2 * p[i] ** 2 / (gamma * E * epsilon[i] ** 2)

    def objective(self, q):
        res = 0
        for i in range(self.num_clients):
            if q[i] < 0:  # q may be smaller than zero due to precision errors of CVX
                return 1e8  # return inf will cause crash, so return a big enough number
            res += self.A[i] / q[i] + self.l * self.B[i] / q[i] ** 0.5

        return res

    def get_q(self):
        q = self.CVXMinimize().x
        print(q)
        return q
