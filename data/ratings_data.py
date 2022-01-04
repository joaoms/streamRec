class RatingsData:

    def __init__(self, user_list: list, item_list: list, ratings_list: list):
        self.userlist = user_list
        self.itemlist = item_list
        self.ratingslist = ratings_list
        self.userset = set(self.userlist)
        self.itemset = set(self.itemlist)
        self.size = len(self.userlist)
        self.BuildMaps()

    def BuildMaps(self):
        self.useritems = {}
        self.itemusers = {}
        for u in self.userset:
            self.useritems[u] = []
        for i in self.itemset:
            self.itemusers[i] = []
        for i in range(self.size):
            self.useritems[self.userlist[i]].append(self.itemlist[i])
            self.itemusers[self.itemlist[i]].append(self.userlist[i])

    def GetUserItems(self, user_id):
        return self.useritems[user_id]

    def GetItemUsers(self, item_id):
        return self.itemusers[item_id]

    def GetRating(self, user_id, item_id):
        idx = self.GetIndex(user_id, item_id)
        return self.ratingslist[idx]

    def GetIndex(self, user_id, item_id):
        i = self.userlist.index(user_id)
        while(self.itemlist[i] != item_id):
            i += 1 + self.userlist[i+1:].index(user_id)

    def AddFeedback(self, user_id, item_id, rating):
        self.userlist.append(user_id)
        self.itemlist.append(item_id)
        self.ratingslist.append(item_id)
        if user_id not in self.userset:
            self.userset.add(user_id)
            self.useritems[user_id] = []
        if item_id not in self.itemset:
            self.itemset.add(item_id)
            self.itemusers[item_id] = []
        self.useritems[user_id].append(item_id)
        self.itemusers[item_id].append(user_id)

    def GetTuple(self, idx: int):
        return self.userlist[idx], self.itemlist[idx]
