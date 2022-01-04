from data import ImplicitData
import numpy as np
from .Model import Model

class ISGD(Model):
    """
    Incremental SGD-based matrix factorization algorithm for implicit feedback:
    Vinagre, J., Jorge, A. M., & Gama, J. (2014, July). Fast incremental matrix factorization for recommendation with positive-only feedback. In International Conference on User Modeling, Adaptation, and Personalization (pp. 459-470). Springer, Cham.
    https://link.springer.com/chapter/10.1007/978-3-319-08786-3_41
    """

    def __init__(self, data: ImplicitData, num_factors: int = 10, num_iterations: int = 10, learn_rate: float = 0.01, u_regularization: float = 0.1, i_regularization: float = 0.1, random_seed: int = 1):
        """
        Constructor.

        Keyword arguments:
        data -- ImplicitData object
        num_factors -- Number of latent features (int, default 10)
        num_iterations -- Maximum number of iterations (int, default 10)
        learn_rate -- Learn rate, aka step size (float, default 0.01)
        regularization -- Regularization factor (float, default 0.01)
        random_seed -- Random seed (int, default 1)
        """
        self.data = data
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.learn_rate = learn_rate
        self.user_regularization = u_regularization
        self.item_regularization = i_regularization
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self._InitModel()


    def _InitModel(self):
        self.ResetModel()

    def ResetModel(self):
        self.user_factors = [np.random.normal(0.0, 0.1, self.num_factors) for _ in range(self.data.maxuserid + 1)]
        self.item_factors = [np.random.normal(0.0, 0.1, self.num_factors) for _ in range(self.data.maxitemid + 1)]

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

    def IncrTrain(self, user, item, update_users: bool = True, update_items: bool = True, n_times: int = 1):
        """
        Incrementally updates the model.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """

        user_id, item_id = self.data.AddFeedback(user, item)

        if len(self.user_factors) == self.data.maxuserid:
            self.user_factors.append(np.random.normal(0.0, 0.1, self.num_factors))
        if len(self.item_factors) == self.data.maxitemid:
            self.item_factors.append(np.random.normal(0.0, 0.1, self.num_factors))

        if update_users or update_items:
            for _ in range(n_times):
                self._UpdateFactors(user_id, item_id, update_users, update_items)

    def _UpdateFactors(self, user_id, item_id, update_users: bool = True, update_items: bool = True, target: int = 1):
        p_u = self.user_factors[user_id]
        q_i = self.item_factors[item_id]
        for _ in range(int(self.num_iterations)):
            err = target - np.inner(p_u, q_i)

            if update_users:
                delta = self.learn_rate * (err * q_i - self.user_regularization * p_u)
                p_u += delta

            if update_items:
                delta = self.learn_rate * (err * p_u - self.item_regularization * q_i)
                q_i += delta

        self.user_factors[user_id] = p_u
        self.item_factors[item_id] = q_i

    def Predict(self, user_id, item_id):
        """
        Return the prediction (float) of the user-item interaction score.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """
        return np.inner(self.user_factors[user_id], self.item_factors[item_id])

    def Recommend(self, user, n: int = -1, exclude_known_items: bool = True, sort_list: bool = True):
        """
        Returns an list of tuples in the form (item_id, score), ordered by score.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """

        user_id = self.data.GetUserInternalId(user)

        if user_id == -1:
            return []

        recs = []

        p_u = self.user_factors[user_id]
        scores = np.abs(1 - np.inner(p_u, self.item_factors))

        recs = np.column_stack((self.data.itemset, scores))

        if exclude_known_items:
            user_items = self.data.GetUserItems(user_id)
            recs = np.delete(recs, user_items, 0)

        if sort_list:
            recs = recs[np.argsort(recs[:, 1], kind = 'heapsort')]

        if n == -1 or n > len(recs) :
            n = len(recs)

        return recs[:n]
