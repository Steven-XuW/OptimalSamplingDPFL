import numpy as np
import json
import multiprocessing
from scipy.special import gammaln, logsumexp
import math


def logcomb(n, k):
    """Returns the logarithm of comb(n,k)"""
    return gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)


class PrivacyAccountant:
    def __init__(self, options, trainer):
        self.trainer = trainer
        self.batchsize = options['batch_size']
        # Meta parameters
        # T: nb of communication rounds
        # K: nb of local updates
        # M: nb of users
        # R: nb of data points used for training

        self.T = options['num_round'] * options['clients_per_round']
        self.K = options['num_epoch']
        self.M = trainer.num_clients
        self.R = int(0.8 * 5000)

        # 0.8 : training ratio

        # delta: privacy parameter
        # l: user subsampling ratio
        # s: data subsampling ratio

        self.delta = trainer.delta
        self.l = 0.2
        self.s = 0.2

        self.sigma = 60
        self.target_epsilon = 16

    def RDP_epsilon_bound_gaussian(self, alpha):
        """Returns the epsilon RDP bound for Gaussian mechanism with std parameter sigma_gaussian_actual"""
        return 0.5 * alpha / (self.sigma ** 2)

    def cgf_subsampling_for_int_alpha(self, alpha: int, eps_func, sub_ratio):
        """
        Parameters:
        :param alpha: int, >1
        :param eps_func: fun(float->float), epsilon RDP bound evaluation function
        :param sub_ratio: subsampling ratio

        Returns a tight upper bound of the CGF(alpha) for the s-subsampled eps_func(alpha),
        ie (alpha-1)*eps_subsampled(alpha)"""
        alpha = int(alpha)
        log_moment_two = 2 * np.log(sub_ratio) + logcomb(alpha, 2) + np.minimum(
            np.log(4) + eps_func(2.) + np.log(1 - np.exp(-eps_func(2.))), eps_func(2.) + np.log(2))
        log_moment_j = lambda j: np.log(2) + (j - 1) * eps_func(j) + j * np.log(sub_ratio) + logcomb(alpha, j)
        all_log_moments_j = [log_moment_j(j) for j in range(3, alpha + 1, 1)]
        return logsumexp([0, log_moment_two] + all_log_moments_j)

    def intermediate_epsilon_rdp_bound_for_int_alpha(self, alpha: int):
        """
        Parameters:
        :param alpha: int, >1

        Returns an upper RDP epsilon bound after K composed s-subsampled Gaussian mechanisms.
        """
        return self.K * self.cgf_subsampling_for_int_alpha(alpha, self.RDP_epsilon_bound_gaussian, self.s) / (alpha - 1)

    def get_optimal_t(self):
        # l = qi * K
        k = 1.70174454109
        return self.l * self.T + math.log(1 / self.delta) * math.sqrt(self.T * self.l * (1 - self.l)) / k

    def epsilon_rdp_bound_for_int_alpha(self, alpha: int):
        """
        Parameters:
        :param alpha: int, >1

        Returns an upper RDP epsilon bound after T composed l-subsampled [K composed s-subsampled Gaussian mechanisms].
        """
        return self.get_optimal_t() * self.cgf_subsampling_for_int_alpha(alpha,
                                                                         self.intermediate_epsilon_rdp_bound_for_int_alpha,
                                                                         1) / (alpha - 1)
        # return T * cgf_subsampling_for_int_alpha(alpha, intermediate_epsilon_rdp_bound_for_int_alpha, l) / (alpha - 1)

    def epsilon_rdp_bound_for_float_alpha(self, alpha: float):
        """
        Parameters:
        :param alpha: float, >1

        Returns an upper RDP epsilon bound after T composed l-subsampled [K composed s-subsampled Gaussian mechanisms],
        using linear interpolation on the CGF (by convexity) to approximate the bound.
        """
        floor_alpha = math.floor(alpha)
        ceil_alpha = math.ceil(alpha)
        if floor_alpha == 1:
            first = 0.
        else:
            first = (1 - alpha + floor_alpha) * (floor_alpha - 1) * self.epsilon_rdp_bound_for_int_alpha(
                floor_alpha) / (
                            alpha - 1)
        second = (alpha - floor_alpha) * (ceil_alpha - 1) * self.epsilon_rdp_bound_for_int_alpha(ceil_alpha) / (
                alpha - 1)
        return first + second

    def epsilon_dp_bound_for_int_alpha(self, alpha: int):
        """
        Parameters:
        :param alpha: int, >1

        Returns an upper DP epsilon bound after T composed l-susampled [K composed s-subsampled Gaussian mechanisms].
        """
        return self.epsilon_rdp_bound_for_int_alpha(alpha) + np.log(1 / self.delta) / (alpha - 1)

    def epsilon_dp_bound_for_float_alpha(self, alpha: float):
        """
        Parameters:
        :param alpha: float, >1

        Returns an upper DP epsilon bound after T composed l-susampled [K composed s-subsampled Gaussian mechanisms].
        """
        return self.epsilon_rdp_bound_for_float_alpha(alpha) + np.log(1 / self.delta) / (alpha - 1)

    def get_epsilon(self):

        # Parameters to tune by hand:
        # alpha_int_max: int
        # n_points: int

        # 1. Determine the integer alpha with the best DP bound (grid search between 2 and alpha_int_max)
        alpha_int_max = 100
        alpha_int_space = np.arange(2, alpha_int_max + 1, 1)
        argmin_int = np.argmin([self.epsilon_dp_bound_for_int_alpha(alpha_int) for alpha_int in alpha_int_space])
        alpha_int_min = alpha_int_space[argmin_int]
        if alpha_int_min == alpha_int_max:
            print("Increase alpha_int_max!")

        alpha_lower = alpha_int_min - 1. + 0.0001  # instability around alpha=1
        alpha_upper = alpha_int_min + 1.

        # 2. Determine the float alpha with the best DP bound (grid search around alpha_int_min: +-1)
        n_points = 1000  # precision of the grid
        alpha_float_space = np.linspace(alpha_lower, alpha_upper, n_points)
        idx_min = np.argmin([self.epsilon_dp_bound_for_float_alpha(alpha_float) for alpha_float in alpha_float_space])
        alpha_float_min = alpha_float_space[idx_min]
        return self.epsilon_dp_bound_for_float_alpha(alpha_float_min)
        # print("Best epsilon DP bound:{:.4f}".format(epsilon_dp_bound_for_float_alpha(alpha_float_min)))

    def binary_search_target(self, target_value, l, r, epsilon):
        left = l
        right = r
        while abs(right - left) > epsilon:
            mid = (left + right) / 2
            self.sigma = mid
            mid_value = self.get_epsilon()
            if mid_value > target_value:
                left = mid
            else:
                right = mid

        return (left + right) / 2

    def get_sigma(self):
        left_bound = 0
        right_bound = 5
        target_value = self.target_epsilon
        epsilon = 5e-2

        sigma_for_all = []
        data_dict_list = []
        for cid, c in enumerate(self.trainer.clients):
            self.R = len(c.train_data)  # R: nb of data points used for training
            # self.delta = 1 / self.R
            self.l = self.trainer.q[cid]
            self.s = self.batchsize / self.R
            target_value = self.trainer.epsilon[cid]
            sigma_for_all.append(self.binary_search_target(target_value, left_bound, right_bound, epsilon))
            print(f'cid is {cid}, sigma is {sigma_for_all[cid]}, epsilon is {target_value}')
            data_dict_list.append({'cid': cid, 'sigma': sigma_for_all[cid], 'epsilon': target_value})

        json_file_path = 'data.json'
        with open(json_file_path, 'w') as json_file:
            json.dump(data_dict_list, json_file, indent=4)

        return sigma_for_all
