import numpy as np

class linear_regression:
    # LinRegr constructor
    def __init__(self, use_intercept=True):
        self.beta = None
        self.use_intercept = use_intercept
    
    def add_intercept(self, x):
        intercepts = np.ones(x.shape[0])
        x = np.column_stack((intercepts, x))
        return x
    
    def fit(self, x, y):
        if self.use_intercept == True:
            x = self.add_intercept(x)
        # Compute the closed Form
        inner = np.dot(x.T, x)
        inv = np.linalg.inv(inner)
        self.beta = np.dot(np.dot(inv, x.T), y)
        
    def predict(self, x):
        if x.shape[1] < self.beta.shape[0] and self.use_intercept == True:
            x = self.add_intercept(x)
        predictions = np.array([np.dot(x_i, self.beta) for x_i in x])
        return predictions
    
    def compute_r2(self, y, y_mean, y_pred):
        frac1 = sum([(y[i] - y_pred[i]) ** 2 for i in range(y.shape[0])])
        frac2 = sum([(y[i] - y_mean) ** 2 for i in range(y.shape[0])])
        r2 = 1 - frac1 / frac2
        return r2
    
    def score(self, x, y):
        y_pred = self.predict(x)
        y_mean = np.mean(y)
        score = self.compute_r2(y, y_mean, y_pred)
        return score
