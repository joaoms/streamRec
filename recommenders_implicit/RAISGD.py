import random
from data import ImplicitData
from .ISGD import ISGD
import numpy as np

class RAISGD(ISGD):
    """
    Recency-adjusted ISGD, as proposed in:
    Vinagre, J., Jorge, A. M., & Gama, J. (2015, April). Collaborative filtering with recency-based negative feedback. In Proceedings of the 30th Annual ACM Symposium on Applied Computing (pp. 963-965).
    https://dl.acm.org/doi/abs/10.1145/2695664.2695998
    """
    def __init__(self, data: ImplicitData, num_factors: int, num_iterations: int, learn_rate: float, regularization: float, random_seed: int, ra_length: int):
        super().__init__(data, num_factors, num_iterations, learn_rate, regularization, random_seed)
        self.ra_length = ra_length

    def _InitModel(self):
        super()._InitModel()
        self.itemqueue = list(self.data.itemset)

    def IncrTrain(self, user_id, item_id, update_users: bool = True, update_items: bool = True):
        if user_id not in self.data.userset:
            self.user_factors[user_id] = np.random.normal(0.0, 0.01, self.num_factors)

        if item_id not in self.data.itemset:
            self.item_factors[item_id] = np.random.normal(0.0, 0.01, self.num_factors)
        else:
            self.itemqueue.remove(item_id)

        self.data.AddFeedback(user_id, item_id)

        for i in range(self.ra_length):
            last = self.itemqueue.pop(0)
            self._UpdateFactors(user_id, last, True, False, 0)
            self.itemqueue.append(last)

        self._UpdateFactors(user_id, item_id)
        self.itemqueue.append(item_id)
