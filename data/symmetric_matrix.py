import numpy as np

class SymmetricMatrix:

    def __init__(self, num_entities: int = 0):
        self.size = (num_entities + 1) * 2
        self.matrix = np.zeros((self.size, self.size))
        self.max_id = num_entities - 1

    def Get(self, i, j):
        if i > self.max_id or i < 0 or j > self.max_id or j < 0:
            return 0
        return self.matrix[i][j]
    
    def GetRow(self, i: int):
        return self.matrix[i][:self.max_id + 1]

    def IncrementDiag(self, i: int):
        if i < 0 or i > self.max_id + 1:
            return False
        if i >= self.max_id + 1:
            if i == self.size:
                self._Resize()
            self.max_id = i
        self.matrix[i][i] += 1
        return True

    def Increment(self, i: int, j: int):
        if i < 0 or j < 0 or i > self.max_id + 1 or j > self.max_id + 1:
            return False
        self.matrix[i][j] += 1
        self.matrix[j][i] += 1
        return True

    def Set(self, i: int, j: int, val: float):
        if i == self.max_id + 1 or j == self.max_id + 1:
            if self.max_id + 2 >= self.size:
                self._Resize()
            self.max_id = max(i, j)
        if i < 0 or j < 0 or i >= self.max_id + 1 or j >= self.max_id + 1:
            return False
        self.matrix[i][j] = val
        self.matrix[j][i] = val
        return True

    def _Resize(self):
        new_size = self.size * 2
        new_matrix = np.zeros((new_size, new_size))
        new_matrix[:self.size,:self.size] = self.matrix
        self.matrix = new_matrix
        self.size = new_size

