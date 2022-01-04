from data import ImplicitData
from recommenders_implicit import *
import time

class EvalLeaveLastOut:

    def __init__(self, model: Model, data: ImplicitData, test_users: list, test_items: list, metrics = ["Recall@20"]):
        # TODO: Input checks
        self.model = model
        self.data = data
        self.model.data = self.data
        self.test_users = test_users
        self.test_items = test_items
        self.metrics = metrics

    def EvaluateTime(self):
        results = dict()
        time_recommend = []
        time_eval_point = []

        for metric in self.metrics:
            results[metric] = []

        start_train = time.time()
        self.model.ResetModel()
        self.model.BatchTrain()
        time_train = time.time() - start_train

        for i in range(len(self.test_users)):

            user = self.test_users[i]
            item = self.test_items[i]

            if item not in self.model.data.GetUserItems(user, False):
                start_recommend = time.time()
                reclist = self.model.Recommend(user, 20)
                end_recommend = time.time()
                time_recommend.append(end_recommend - start_recommend)

                start_eval_point = time.time()
                results[metric].append(self.__EvalPoint(item, reclist))
                end_eval_point = time.time()
                time_eval_point.append(end_eval_point - start_eval_point)


        results['time_train'] = time_train
        results['time_recommend'] = time_recommend
        results['time_eval_point'] = time_eval_point

        return results

    def Evaluate(self, start_eval = 0, count = 0, interleaved = 1):
        results = dict()

        if not count:
            count = self.data.size

        count = min(count, self.data.size)

        for metric in self.metrics:
            results[metric] = []

        for i in range(count):
            uid, iid = self.data.GetTuple(i)
            if i >= start_eval and i % interleaved == 0 and iid not in self.model.data.GetUserItems(uid, False):
                reclist = self.model.Recommend(uid)
                results[metric].append(self.__EvalPoint(iid, reclist))
            self.model.IncrTrain(uid, iid)

        return results

    def __EvalPoint(self, item_id, reclist):
        result = 0
        if len(reclist) == 0:
            return 0
        for metric in self.metrics:
            if metric == "Recall@20":
                #print('reclist', reclist)
                #print('len(reclist)', len(reclist))
                #reclist = [x[0] for x in reclist[:20]]
                result = int(item_id in reclist[:20,0])
        return result
