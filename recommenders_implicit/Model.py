
class Model:

    def __init__(self):
        pass

    def BatchTrain(self):
        """
        Trains a new model with the available data.
        """
        pass

    def IncrTrain(self, user_id, item_id, update_users: bool = True, update_items: bool = True):
        """
        Incrementally updates the model.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """
        pass

    def Predict(self, user_id, item_id):
        """
        Return the prediction (float) of the user-item interaction score.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """
        pass

    def Recommend(self, user_id: int, n: int = -1, candidates: set = {}, exclude_known_items: bool = True):
        """
        Returns an list of tuples in the form (item_id, score), ordered by score.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """
        pass
