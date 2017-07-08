import numpy as np

class Perceptron():
    def __init__(self):
        self.w = np.zeros(3, dtype=float)
        
    def predict(self, X):
        XX = np.hstack((np.ones(len(X)).reshape((-1, 1)), X))
        y = np.sign(np.dot(XX, self.w))
        return y
    
    def fit(self, X, y, initial_weights=np.zeros(3, dtype=float), collect_weights=False):
        self.w = initial_weights
        weights = []
        steps = 0
        while(True):
            if collect_weights:
                weights.append(self.w.copy())
            else:
                steps += 1
                
            y_ = self.predict(X)
            err_idx = y_ != y
            
            if sum(err_idx) == 0:
                break
                
            errX = X[err_idx]
            errY = y[err_idx]
            rand_idx = np.random.randint(0, len(errX))
            self.adjust_weights(errX[rand_idx], errY[rand_idx])
            
        return np.array(weights) if collect_weights else steps
    
        
    def adjust_weights(self, x, y):
        self.w += np.concatenate(([1], x)) * y 