from collections import defaultdict
import random

from numpy.core.numeric import NaN
from data import ImplicitData
import numpy as np
from .Model import Model

class BISGD(Model):
    def __init__(self, data: ImplicitData, num_factors: int = 10, num_iterations: int = 10, num_nodes: int = 5, learn_rate: float = 0.01, u_regularization: float = 0.1, i_regularization: float = 0.1, random_seed: int = 1):
        """    Constructor.

        Keyword arguments:
        data -- ImplicitData object
        num_factors -- Number of latent features (int, default 10)
        num_iterations -- Maximum number of iterations (int, default 10)
        learn_rate -- Learn rate, aka step size (float, default 0.01)
        regularization -- Regularization factor (float, default 0.01)
        random_seed -- Random seed (int, default 1)"""

        self.counter=0
        self.data = data
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.learn_rate = learn_rate
        self.user_regularization = u_regularization
        self.item_regularization = i_regularization
        self.random_seed = random_seed
        self.num_nodes = num_nodes
        np.random.seed(random_seed)
        self._InitModel()

    def _InitModel(self):
        self.user_factors = [[np.random.Generator.normal(0.0, 0.1, self.num_factors) for _ in range(self.data.maxuserid + 1)] for _ in range(self.num_nodes)]
        self.item_factors = [[np.random.Generator.normal(0.0, 0.1, self.num_factors) for _ in range(self.data.maxitemid + 1)] for _ in range(self.num_nodes)]


    def BatchTrain(self):
        """
        Trains a new model with the available data.
        """
        idx = list(range(self.data.size))
        for iter in range(self.num_iterations):
            np.random.shuffle(idx)
            for i in idx:
                user_id, item_id = self.data.GetTuple(i, True)
                self._UpdateFactors(user_id, item_id)

    def IncrTrain(self, user, item, update_users: bool = True, update_items: bool = True):
        """
        Incrementally updates the model.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """

        user_id, item_id = self.data.AddFeedback(user, item)

        for node in range(self.num_nodes):
            if len(self.user_factors[node]) == self.data.maxuserid:
                self.user_factors[node].append(np.random.normal(0.0, 0.1, self.num_factors))
        for node in range(self.num_nodes):
            if len(self.item_factors[node]) == self.data.maxitemid:
                self.item_factors[node].append(np.random.normal(0.0, 0.1, self.num_factors))


        for node in range(self.num_nodes):
            kappa = min(1, int(np.random.poisson(1, size=1)))

            if kappa > 0:
                for _ in range(kappa):
                    self._UpdateFactors(user_id, item_id, node)


    def _UpdateFactors(self, user_id, item_id, node, update_users: bool = True, update_items: bool = True, target: int = 1):

        p_u = self.user_factors[node][user_id]
        q_i = self.item_factors[node][item_id]

        for _ in range(int(self.num_iterations)):
            err = target - np.inner(p_u, q_i)

            if update_users:
                delta = self.learn_rate * (err * q_i - self.user_regularization * p_u)
                p_u += delta

            if update_items:
                delta = self.learn_rate * (err * p_u - self.item_regularization * q_i)
                q_i += delta

        self.user_factors[node][user_id] = p_u
        self.item_factors[node][item_id] = q_i

    def Predict(self, user_id, item_id):
        """
        Return the prediction (float) of the user-item interaction score.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """
        rec = 0
        for node in range(self.num_nodes):
            rec = rec + np.inner(self.user_factors[node][user_id], self.item_factors[node][item_id])
        return rec / self.num_nodes

    def Recommend(self, user, n: int = -1, exclude_known_items: bool = True):

        user_id = self.data.GetUserInternalId(user)
        if user_id == -1:
            return []
        
        if exclude_known_items:
            user_items = self.data.GetUserItems(user_id)

        precs = [None for _ in range(self.num_nodes)]
        for node in range(self.num_nodes):
            p_u = self.user_factors[node][user_id]
            scores = np.abs(1 - np.inner(p_u, self.item_factors[node]))
            recs_node = np.column_stack((self.data.itemset, scores))
            
            if exclude_known_items:
                recs_node = np.delete(recs_node, user_items, 0)
            
            if n == -1 or n > len(recs_node) :
                n = len(recs_node)

            # recs_node = recs_node[np.argsort(recs_node[:, 1], kind = 'heapsort')]
            # recs_node = recs_node[:n]
            # No need to sort yet, just partition:
            recs_node = recs_node[np.argpartition(recs_node[:,1], n-1)[:n]]
            
            precs[node] = recs_node

        recs = self._AvgRecs(precs)

        recs = recs[np.argsort(-recs[:, 1].astype(np.float), kind = 'heapsort')]

        if n == -1 or n > len(recs):
            n = len(recs)

        return recs[:n]

    def _AvgRecs(self, precs):
        recs_dict = {}
        for node in range(self.num_nodes):
            for pair in precs[node]:
                if pair[0] in recs_dict:
                    recs_dict[pair[0]] = recs_dict[pair[0]] + 1 - pair[1].astype(np.float)
                else:
                    recs_dict[pair[0]] = 1 - pair[1].astype(np.float)
        
        return np.array([[i, recs_dict[i]/self.num_nodes] for i in recs_dict])
 
    def RecommendOld(self, user, n: int = -1, exclude_known_items: bool = True):

        user_id = self.data.GetUserInternalId(user)
        if user_id == -1:
            return []


        recommendation_list = np.zeros((self.data.maxitemid + 1, self.num_nodes)) + 1

        for node in range(self.num_nodes):
            p_u = self.user_factors[node][user_id]
            
            recommendation_list[:,node] = np.abs(1 - np.inner(p_u, self.item_factors[node]))

        scores = np.mean(recommendation_list, 1)
        recs = np.column_stack((self.data.itemset, scores))

        if exclude_known_items:
            user_items = self.data.GetUserItems(user_id)
            recs = np.delete(recs, user_items, 0)

        recs = recs[np.argsort(recs[:, 1], kind = 'heapsort')]

        if n == -1 or n > len(recs) :
            n = len(recs)

        return recs[:n]

