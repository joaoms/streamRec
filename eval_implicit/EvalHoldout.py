from data import ImplicitData
from recommenders_implicit import *
import numpy as np
import pandas as pd
import time
import random

class EvalHoldout:
    # TODO: Documentation

    def __init__(self, model: Model, holdout: ImplicitData, metrics = ["Recall@20"]):
        # TODO: Input checks
        self.model = model
        self.holdout = holdout
        self.metrics = metrics


    def EvaluateTime(self):
        results = dict()
        time_get_tuple = []
        time_recommend = []
        time_eval_point = []

        for metric in self.metrics:
            results[metric] = []

        for i in range(self.holdout.size):
            start_get_tuple = time.time()
            uid, iid = self.holdout.GetTuple(i)
            end_get_tuple = time.time()
            time_get_tuple.append(end_get_tuple - start_get_tuple)

            if iid not in self.model.data.GetUserItems(uid, False):
                start_recommend = time.time()
                reclist = self.model.Recommend(uid, 20)
                end_recommend = time.time()
                time_recommend.append(end_recommend - start_recommend)

                start_eval_point = time.time()
                results[metric].append(self.__EvalPoint(iid, reclist))
                end_eval_point = time.time()
                time_eval_point.append(end_eval_point - start_eval_point)


        results['time_get_tuple'] = time_get_tuple
        results['time_recommend'] = time_recommend
        results['time_eval_point'] = time_eval_point

        return results

    def Evaluate(self):
        results = dict()

        for metric in self.metrics:
            results[metric] = []

        for i in range(self.holdout.size):
            uid, iid = self.holdout.GetTuple(i)

            if iid not in self.model.data.GetUserItems(uid, False):
                reclist = self.model.Recommend(uid, 20)
                results[metric].append(self.__EvalPoint(iid, reclist))

        return results

    def __EvalPoint(self, item_id, reclist):
        result = 0
        if len(reclist) == 0:
            return 0
        for metric in self.metrics:
            if metric == "Recall@20":
                result = int(item_id in reclist[:20,0])
        return result
