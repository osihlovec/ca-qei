from botorch.models.gp_regression import FixedNoiseGP
import torch
from scipy.stats.qmc import LatinHypercube
import numpy as np
from utils import chebyshev_scalar, chebyshev_scalar_batch, rescale, calculate_cost


class ExpectedImprovementCalculator():
    def __init__(self, observed_x, observed_y1, observed_y2, max_val, min_x, max_x, reference, q = 8, iterations = 30, epochs = 50, restarts = 10, 
                 mc = 10, noise = 0.00001, alpha = 1, gamma = 2, cost_aware = True, weights = None, utilities = torch.tensor([0.5, 0.5]), version = 1, 
                 device = torch.device('cpu'), misspecified = False, seed = 0, normalization = 10):
        self.max_val = max_val.to(torch.double) # most optimal input thus far
        self.min_x = min_x # lower bound of the input space per dimension
        self.max_x = max_x # upper bound of the input space per dimension
        self.reference = reference.to(torch.double) # reference point (reservation utilities) for chebyshev scalarization
        self.q = q # batch size (degree of parallelism)
        self.iterations = iterations # number of iterations (samples) used to estimate the gradient
        self.epochs = epochs # number of optimizer (ADAM) steps
        self.restarts = restarts # number of initializations to avoid local optima
        self.mc = mc # number of monte carlo epochs to estimate expected gradient
        self.noise = noise # distortion of the objective output 
        self.alpha = alpha # currently unusued, but scales the gradient
        self.gamma = gamma # currently unusued, but can be used to regularize the gradient
        self.cost_aware = cost_aware # controls between cost-aware and cost-agnostic
        self.sampler = LatinHypercube(d = observed_x.size(dim = 1), seed = seed) # sampler of initial points (restarts)
        self.observed_x = observed_x.to(torch.double) # tensor of previously queried datapoints
        self.model1 = FixedNoiseGP(observed_x, observed_y1, torch.full_like(observed_y1, noise)) # GP of the first objective function
        self.model2 = FixedNoiseGP(observed_x, observed_y2, torch.full_like(observed_y2, noise)) # GP of the second objective function
        self.weights = weights # weights (relative costs) of the input dimensions
        self.utilities = utilities # utilities of the two objectives that shape the final (scalarized) utility
        self.version = version # switches the internal optimizer
        self.device = device # GPU or CPU
        self.misspecified = misspecified # standard or delayed environment
        self.seed = seed
        self.normalization = normalization # normalization constant for the cost function
    
    def rbf_covariance(self, x, y):
        # calculates the covariance matrix
        k_xx = torch.exp((torch.linalg.norm((x.unsqueeze(1)-x), dim=2, ord=2) ** 2) / (-2))
        k_xy = torch.exp((torch.linalg.norm((x.unsqueeze(1)-y), dim=2, ord=2) ** 2) / (-2))
        k_yy = torch.exp((torch.linalg.norm((y.unsqueeze(1)-y), dim=2, ord=2) ** 2) / (-2))
        
        return k_xx - k_xy @ torch.inverse(k_yy) @ k_xy.T
    
    def function_h(self, x):
        # returns the improvement of the selected input over the best-found datapoint
        
         # compute means
         m1 = self.model1.posterior(x)     
         m2 = self.model2.posterior(x)
         
         # compute cholesky decomposition
         cov_matrix = self.rbf_covariance(rescale(x, self.min_x, self.max_x), rescale(self.observed_x, self.min_x, self.max_x))
         l =  torch.cholesky(cov_matrix + self.noise * torch.eye(len(x), dtype = torch.double, device = self.device))
         
         if (self.misspecified):
             res = torch.max(chebyshev_scalar(m1.mean + l @ self.z, m2.mean + l @ self.z, self.reference, self.utilities) - self.max_val)
         else:
             input_x = x = torch.transpose(torch.stack((m1.mean + l @ self.z, m2.mean + l @ self.z), dim = 1).repeat(1, 1, len(x)), 1, 2)
             res = torch.max(chebyshev_scalar_batch(input_x, self.reference, self.utilities) - self.max_val)
         return torch.clamp(res, 0)
     
    def estimate_gradient(self, x, t = 0):
        accumulator = torch.zeros(x.size(), dtype = torch.double, device = self.device)
        for i in range(self.iterations):
            self.z = torch.normal(torch.zeros((x.size()[0], 1), dtype = torch.double, device = self.device), 
                                  torch.ones((x.size()[0], 1), dtype = torch.double, device = self.device))
            if (self.cost_aware):
                cost = torch.sum(calculate_cost(x, self.weights, t, self.normalization) * (self.function_h(x))) / len(x)
                #print(torch.sum(x, 0) / len(x))
                grad = torch.autograd.grad(cost, x)[0]
                accumulator += grad
            else:
                grad = torch.autograd.grad(self.function_h(x), x)[0]
                #print(grad)
                accumulator += grad
        
        return accumulator / self.iterations
    
    def estimate_stationary(self, x, t = 0):
        # returns the saddlepoint of the given optimization subroutine
        
        accumulator = [x]
        previous = x.detach()
        previous.requires_grad = True
        for i in range(self.epochs):
            grad = self.estimate_gradient(previous, t)
            previous = previous + grad
            #print(previous)
            previous = torch.clamp(previous, self.min_x, self.max_x)
            accumulator.append(previous)
        
        accumulator = torch.stack(accumulator, dim = 0)
        stationary_points = torch.sum(accumulator, axis = 0) / (self.epochs + 1)
        
        return stationary_points
    
    def estimate_gradient2(self, x, t = 0):
        cost = torch.tensor([0.0], dtype = torch.double, device = self.device)
        for i in range(self.iterations):
            self.z = torch.normal(torch.zeros((x.size()[0], 1), dtype = torch.double, device = self.device), 
                                  torch.ones((x.size()[0], 1), dtype = torch.double, device = self.device))
            if (self.cost_aware):
                cost += torch.sum(calculate_cost(x, self.weights, t, self.max_x[0]) * (self.function_h(x))) / len(x)
            else:
                cost += (self.function_h(x))
        
        cost /= self.iterations
        return cost
    
    def estimate_stationary2(self, x, t = 0):
        x_copy = x.detach()
        x_copy.requires_grad = True
        self.lr = 0.0333
        self.weight_decay = 5e-4
        optimizer = torch.optim.Adam([x_copy], lr=self.lr, weight_decay=self.weight_decay)
        
        for i in range(self.epochs):
            optimizer.zero_grad()
            expectation = -self.estimate_gradient2(x_copy, t)
            expectation.backward(retain_graph = False)
            optimizer.step()
            with torch.no_grad():
                x_copy[:] = x_copy.clamp(self.min_x, self.max_x)
        
        x_copy.requires_grad = False
        return x_copy
         
    def getNext(self, t = 0, n = None):
        # outputs the next query point
        # time controls the cost function
        # if n is specified, then the result batch will have size of n
        best_point = None
        best_ei = None
        
        for i in range(self.restarts):
            if (n == None):
                latin = self.sampler.random(n = self.q)
            else:
                latin = self.sampler.random(n = n)
            latin = rescale(torch.from_numpy(latin).to(dtype = torch.double, device = self.device), self.min_x, self.max_x)
            
            if (self.version == 1):
                stat_point = self.estimate_stationary(latin, t)
            else:
                stat_point = self.estimate_stationary2(latin, t)
            ei = 0
            
            for j in range(self.mc):
                self.z = torch.normal(torch.zeros((stat_point.size()[0], 1), dtype = torch.double, device = self.device), 
                                      torch.ones((stat_point.size()[0], 1), dtype = torch.double, device = self.device))
                ei += self.function_h(stat_point)
            
            ei /= self.mc
            
            if (best_point == None or best_ei < ei):
                best_point = stat_point
                best_ei = ei
        
        return best_point
    
    def predict(self, x):
        # outputs the inferred objective value of the two objective functions
        # used for delayed environment imputation
        
        m1 = self.model1.posterior(x)
        m2 = self.model2.posterior(x)
        
        y1 = torch.normal(m1.mean, m1.variance)
        y2 = torch.normal(m2.mean, m2.variance)
        
        return (y1, y2)