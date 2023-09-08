import multiprocessing
from src.utils.get_epsilon_bound import PrivacyAccountant


class ParallelOptimizer:

    def __init__(self, trainer, options):
        self.trainer = trainer
        self.options = options

        # 使用Manager创建Pool
        self.manager = multiprocessing.Manager()
        self.pool = self.manager.Pool(processes=4)

    def get_client_sigma(self, cid, c):
        # 复用PrivacyAccountant的逻辑
        accountant = PrivacyAccountant(self.options, self.trainer)
        # 设置参数
        accountant.R = len(c.train_data)
        accountant.l = self.trainer.q[cid]
        accountant.s = self.options['batch_size'] / accountant.R
        accountant.target_epsilon = self.trainer.epsilon[cid]
        # 计算
        sigma = accountant.binary_search_target(accountant.target_epsilon, 0, 10, 5e-2)
        return sigma

    def run(self):
        results = self.pool.map(self.get_client_sigma, [(cid, c) for cid, c in enumerate(self.trainer.clients)])
        self.pool.close()
        self.pool.join()
        return results
