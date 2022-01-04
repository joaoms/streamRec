from numba import njit
from collections import defaultdict
import random
from data import ImplicitData
import numpy as np
from .BISGD import BISGD
from .ISGD import ISGD

class LocalUBISGD(BISGD):
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
        #self.metamodel = ISGD(self.data, self.num_nodes, self.num_iterations, self.learn_rate, self.user_regularization, self.item_regularization, random_seed=self.random_seed)
        self.metamodel_users = [np.abs(np.random.normal(0.0, 0.1, self.num_nodes)) for _ in range(self.data.maxuserid + 1)]
        self.metamodel_items = [np.abs(np.random.normal(0.0, 0.1, self.num_nodes)) for _ in range(self.data.maxuserid + 1)]


    def IncrTrain(self, user, item, update_users: bool = True, update_items: bool = True):
        """
        Incrementally updates the model.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """

        user_id, item_id = self.data.AddFeedback(user, item)

        #self.metamodel.IncrTrain(user, item)

        if len(self.user_factors[0]) == self.data.maxuserid:
            self.metamodel_users.append(np.abs(np.random.normal(0.0, 0.1, self.num_nodes)))
            for node in range(self.num_nodes):
                self.user_factors[node].append(np.random.normal(0.0, 0.1, self.num_factors))
        if len(self.item_factors[0]) == self.data.maxitemid:
            self.metamodel_items.append(np.abs(np.random.normal(0.0, 0.1, self.num_nodes)))
            for node in range(self.num_nodes):
                self.item_factors[node].append(np.random.normal(0.0, 0.1, self.num_factors))
        
        self._UpdateFactorsMeta(user_id, item_id)
        user_vector = self.metamodel_users[user_id]

        for node in np.argsort(-user_vector)[:int(np.round(self.num_nodes*(1-0.368)))]:
            self._UpdateFactors(user_id, item_id, node)

    def _UpdateFactorsMeta(self, user_id, item_id, update_users: bool = True, update_items: bool = True, target: int = 1):
        p_u = self.metamodel_users[user_id]
        q_i = self.metamodel_items[item_id]
        for _ in range(int(self.num_iterations)):
            err = target - np.inner(p_u, q_i)

            if update_users:
                delta = self.learn_rate * (err * q_i - self.user_regularization * p_u)
                p_u += delta
                p_u[p_u<0] = 0.0

            if update_items:
                delta = self.learn_rate * (err * p_u - self.item_regularization * q_i)
                q_i += delta
                q_i[q_i<0] = 0.0

        self.metamodel_users[user_id] = p_u
        self.metamodel_items[item_id] = q_i

