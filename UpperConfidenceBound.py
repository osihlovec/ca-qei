from botorch.models.gp_regression import FixedNoiseGP
import torch
from scipy.stats.qmc import LatinHypercube
from utils import chebyshev_scalar, rescale, calculate_cost

class UpperConfidenceBoundCalculator():
    def __init__(self, observed_x, observed_y1, observed_y2, min_val, min_x, max_x, reference, noise = 0.0001, 
                 beta = 1.2, lr = 0.01, weight_decay=5e-4, restarts = 10, cost_aware = True, weights = None, utilities = torch.tensor([0.5, 0.5]), device = None, seed = 0, normalization = 10):
        self.min_val = min_val
        self.min_x = min_x
        self.max_x = max_x
        self.reference = reference
        self.noise = noise
        self.beta = beta
        self.lr = lr
        self.weight_decay = weight_decay
        self.restarts = restarts
        self.cost_aware = cost_aware
        self.sampler = LatinHypercube(d = observed_x.size(dim = 1), seed = seed)
        self.model1 = FixedNoiseGP(observed_x, observed_y1, torch.full_like(observed_y1, noise))
        self.model2 = FixedNoiseGP(observed_x, observed_y2, torch.full_like(observed_y2, noise))
        self.weights = weights
        self.utilities = utilities
        self.normalization = normalization
    
    def function_h(self, x):
        m1 = self.model1.posterior(x)
        m2 = self.model2.posterior(x)
        return chebyshev_scalar(m1.mean + self.beta * m1.variance, m2.mean + self.beta * m2.variance, self.reference, self.utilities)
         
    def getNext(self, time = 0):
        
        latin = self.sampler.random(n = self.restarts)
        latin = rescale(torch.from_numpy(latin).type(torch.double), self.min_x, self.max_x)
        best = -1
        best_x = latin[0].view(1, 2).detach()
        
        for j in range(self.restarts):
            x = latin[j].view(1, 2).detach()
            x.requires_grad = True
            optimizer = torch.optim.Adam([x], lr=self.lr, weight_decay=self.weight_decay)
            
            for i in range(100):
                optimizer.zero_grad()
                if (self.cost_aware):
                    cost = calculate_cost(x, self.weights, time, self.normalization)
                    loss = -(self.function_h(x) * cost)
                else:
                    loss = -(self.function_h(x))
                loss = loss[0]
                loss.backward(retain_graph = True)
                optimizer.step()
                with torch.no_grad():
                    x[:] = x.clamp(self.min_x[0], self.max_x[0])
            
            if (best < loss):
                best = loss
                best_x = x
        
        return best_x
    
    def predict(self, x):
        m1 = self.model1.posterior(x)
        m2 = self.model2.posterior(x)
        
        y1 = torch.normal(m1.mean, m1.variance)
        y2 = torch.normal(m2.mean, m2.variance)
        
        return (y1, y2)