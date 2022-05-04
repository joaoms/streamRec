import random
from data import ImplicitData
from .ISGD import ISGD
import numpy as np

class RSISGD(ISGD):
    """
    
    """
    def __init__(self, data: ImplicitData, num_factors: int, num_iterations: int, learn_rate: float, regularization: float, random_seed: int, ra_length: int = 1):
        super().__init__(data, num_factors, num_iterations, learn_rate, regularization, random_seed)
        self.ra_length = ra_length

    def IncrTrain(self, user, item, update_users: bool = True, update_items: bool = True):
        user_id, item_id = self.data.AddFeedback(user, item)

        user_items = self.data.GetUserItems(user_id)
        for _ in range(self.ra_length):
            negative_item_id = random.choice(self.data.items)
            while negative_item_id in user_items:
                negative_item_id = random.choice(self.data.items)

            self._UpdateFactors(user_id, negative_item_id, target = 0)
