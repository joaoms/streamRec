from numba import njit
from collections import defaultdict
import random
from data import ImplicitData
import numpy as np
from .BISGD import BISGD

class UBISGD(BISGD):
    def __init__(self, data: ImplicitData, num_factors: int = 10, num_iterations: int = 10, num_nodes: int = 8, learn_rate: float = 0.01, u_regularization: float = 0.1, i_regularization: float = 0.1, random_seed: int = 1):
        """    Constructor.

        Keyword arguments:
        data -- ImplicitData object
        num_factors -- Number of latent features (int, default 10)
        num_iterations -- Maximum number of iterations (int, default 10)
        num_nodes -- Number of 'weak' learners (int, default 8)
        learn_rate -- Learn rate, aka step size (float, default 0.01)
        regularization -- Regularization factor (float, default 0.01)
        random_seed -- Random seed (int, default 1)"""

        super().__init__(data, num_factors, num_iterations, num_nodes, learn_rate, u_regularization, i_regularization, random_seed)

    def _InitModel(self):
        super()._InitModel()
        self.user_k = [[] for _ in range(self.num_nodes)]

    def IncrTrain(self, user, item, update_users: bool = True, update_items: bool = True):
        """
        Incrementally updates the model.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """

        user_id, item_id = self.data.AddFeedback(user, item)

        if len(self.user_factors[0]) == self.data.maxuserid:
            for node in range(self.num_nodes):
                self.user_k[node].append(np.random.poisson(1, size=1)[0])
                self.user_factors[node].append(np.random.normal(0.0, 0.1, self.num_factors))
        if len(self.item_factors[0]) == self.data.maxitemid:
            for node in range(self.num_nodes):
                self.item_factors[node].append(np.random.normal(0.0, 0.1, self.num_factors))
        
        for node in range(self.num_nodes):
            k = min(1, self.user_k[node][user_id])

            if k > 0:
                for _ in range(k):
                    self._UpdateFactors(user_id, item_id, node)



