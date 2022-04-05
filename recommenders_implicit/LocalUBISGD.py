from numba import njit
from collections import defaultdict
import random
from data import ImplicitData
import numpy as np
from .BISGD import BISGD
from .ISGD import ISGD

class LocalUBISGD(BISGD):
    def __init__(self, data: ImplicitData, 
        num_clusters: int = 10, cl_num_iterations: int = 10, cl_learn_rate: float = 0.01, cl_regularization: float = 0.1,         
        num_factors: int = 10, num_iterations: int = 10, learn_rate: float = 0.01, regularization: float = 0.1, 
        random_seed: int = 1):
        """    Constructor.

        Keyword arguments:
        data -- ImplicitData object
        num_clusters -- Number of clusters (int, default 10)
        cl_num_iterations -- Number of iterations of the clustering algorithm (int, default 10)
        cl_learn_rate -- Learn rate of the clustering algorithm (float, default 0.1)
        cl_regularization -- Regularization factor of the clustering algorithm (float, default 0.1)
        num_factors -- Number of latent features (int, default 10)
        num_iterations -- Maximum number of iterations (int, default 10)
        learn_rate -- Learn rate, aka step size (float, default 0.01)
        regularization -- Regularization factor (float, default 0.01)
        random_seed -- Random seed (int, default 1)"""

        self.cl_num_iterations = cl_num_iterations
        self.cl_learn_rate = cl_learn_rate
        self.cl_regularization = cl_regularization

        super().__init__(data, num_factors, num_iterations, num_clusters, learn_rate, regularization, regularization, random_seed)

    def _InitModel(self):
        super()._InitModel()
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
        for _ in range(int(self.cl_num_iterations)):
            err = target - np.inner(p_u, q_i)

            if update_users:
                delta = self.cl_learn_rate * (err * q_i - self.cl_regularization * p_u)
                p_u += delta
                p_u[p_u<0] = 0.0

            if update_items:
                delta = self.cl_learn_rate * (err * p_u - self.cl_regularization * q_i)
                q_i += delta
                q_i[q_i<0] = 0.0

        self.metamodel_users[user_id] = p_u
        self.metamodel_items[item_id] = q_i

