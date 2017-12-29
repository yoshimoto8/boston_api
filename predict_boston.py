import numpy as np
from sklearn.externals  import joblib

class PredictBostonData(object):
    def __init__(self, lstat, crim, age):
        self.data = np.array([lstat, crim, age]).reshape(1,3)

    def predict(self):
        clf = joblib.load('linear.pkl')
        return clf.predict(self.data)
