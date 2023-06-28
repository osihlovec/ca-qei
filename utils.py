import torch
from torch.distributions.dirichlet import Dirichlet

def chebyshev_scalar(x1, x2, reference, utilities = torch.tensor([0.5, 0.5])):
    return torch.min(torch.cat((x1, x2), dim = 1) * utilities.T - reference, dim = 1)[0]

def chebyshev_scalar_batch(x, reference, utilities = torch.tensor([0.5, 0.5])):
    # used for delayed experiments
    return torch.min(x * utilities - reference, dim = 1)[0]

def generate_weights(dimension):
    # generates the weights (relative costs) of the input dimensions
    m = Dirichlet(torch.ones(dimension))
    weights = m.sample()
    weights, _ = torch.sort(weights, descending = True)
    return weights

def calculate_cost(x, weights, t, normalization = 10):
    # cost heuristic
    lambd = 1/(weights * t + 1)
    costs = lambd * torch.exp(-lambd * abs(x / normalization))
    return torch.prod(costs, 1)

def rescale(x, minimums, maximums):
    # if x is normalized, then the output is scaled into range [minimums, maximums]
    return minimums + x * (maximums - minimums)

def normalize(x, minimum, maximum):
    # normalizes the input into range [0, 1]
    return (x - minimum) / (maximum - minimum)