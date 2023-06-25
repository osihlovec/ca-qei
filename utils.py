import torch
from torch.distributions.dirichlet import Dirichlet

def chebyshev_scalar(x1, x2, reference, utilities = torch.tensor([0.5, 0.5])):
    # might have to redefine this to take weights and multiple objectives
    # into account  
    #print(torch.min(torch.cat((x1, x2), dim = 1) * utilities.T - reference, dim = 1)[0])
    return torch.min(torch.cat((x1, x2), dim = 1) * utilities.T - reference, dim = 1)[0]

def chebyshev_scalar_batch(x, reference, utilities = torch.tensor([0.5, 0.5])):
    # might have to redefine this to take weights and multiple objectives
    # into account  
    #print(torch.min(torch.cat((x1, x2), dim = 1) * utilities.T - reference, dim = 1)[0])
    #print(torch.cat((x1, x2), dim = 0).T)
    return torch.min(x * utilities - reference, dim = 1)[0]

def generate_weights(dimension):
    m = Dirichlet(torch.ones(dimension))
    weights = m.sample()
    weights, _ = torch.sort(weights, descending = True)
    return weights

def calculate_cost(x, weights, t, normalization = 10):
    lambd = 1/(weights * t + 1)
    costs = lambd * torch.exp(-lambd * abs(x / normalization))
    return torch.prod(costs, 1)
    # x = x.to(torch.double)
    # costs = (1 - (abs(x)/10) ** (t+1)) @ weights.T
    # return costs.to(torch.double)

def rescale(x, minimums, maximums):
    return minimums + x * (maximums - minimums)

def normalize(x, minimum, maximum):
    return (x - minimum) / (maximum - minimum)