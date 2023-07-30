import numpy as np

class Metrics():
    def __init__(self):
        pass

    def Mean_Squared_Error(y1,y2):
        return np.mean(np.sum((y1-y2)**2))

    def Mean_Absolute_Error(y1, y2):
        return np.mean(np.sum(np.abs(y1-y2)))