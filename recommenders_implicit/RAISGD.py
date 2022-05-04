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
    def __init__(self, data: ImplicitData, num_factors: int = 10, num_iterations: int = 10, learn_rate: float = 0.01, u_regularization: float = 0.1,    i_regularization: float = 0.1, random_seed: int = 1, ra_length: int = 1):
        super().__init__(data, num_factors, num_iterations, learn_rate, u_regularization, i_regularization, random_seed)
        self.ra_length = ra_length

    def _InitModel(self):
        super()._InitModel()
        self.itemqueue = list(self.data.itemset)

    def IncrTrain(self, user, item, update_users: bool = True, update_items: bool = True):
        user_id, item_id = self.data.AddFeedback(user, item)

        if len(self.user_factors) == self.data.maxuserid:
            self.user_factors.append(np.random.normal(0.0, 0.1, self.num_factors))
        if len(self.item_factors) == self.data.maxitemid:
            self.item_factors.append(np.random.normal(0.0, 0.1, self.num_factors))
        else:
            self.itemqueue.remove(item_id)


        if len(self.itemqueue):
            for _ in range(self.ra_length):
                last = self.itemqueue.pop(0)
                self._UpdateFactors(user_id, last, True, False, 0)
                self.itemqueue.append(last)

        self._UpdateFactors(user_id, item_id)
        self.itemqueue.append(item_id)
