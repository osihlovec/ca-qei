# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.
import numpy as np
from tqdm import tqdm
from random import seed
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from utils import generate_weights, chebyshev_scalar, chebyshev_scalar_batch, rescale, calculate_cost, normalize
from ParallelExpectedImprovement import ExpectedImprovementCalculator
from UpperConfidenceBound import UpperConfidenceBoundCalculator
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
import matplotlib.pyplot as plt

# BENCHMARK 1 func 1
def objective2(x):
    # cross in tray
    return -0.00001 * (abs(torch.sin(x[0]) * torch.sin(x[1]) * torch.exp(abs(100 - torch.sqrt(x[0] ** 2 + x[1] ** 2) / torch.pi))) + 1) ** 0.1

# BENCHMARK 1 func 2
def objective3(x):
    # holder table function
    return (-1) * (abs(torch.sin(x[0]) * torch.cos(x[1]) * torch.exp(abs(1 - torch.sqrt(x[0] ** 2 + x[1] ** 2) / torch.pi))))

# BENCHMARK 2 func 1
def objective4(x):
    # shifted drop-wave function
    return 2 * (1 + torch.cos(12 * torch.sqrt(torch.square(x[0] + torch.pi) + torch.square(x[1] - torch.pi)))) / (torch.square(x[0] + torch.pi) 
                                                                                                                  + torch.square(x[1] - torch.pi) + 4)

# BENCHMARK 2 func 2
def objective5(x):
    # 2d rastrigin function
    return 20 + torch.square(x[0]) + torch.square(x[1]) - 10 * (torch.cos(2 * torch.pi * x[0]) + torch.cos(2 * torch.pi * x[1]))
    
def test2d(acquisition_function = "qEI"):
    #┘ optimization of cross-in-tray and holder table functions in correct environment (with immediate feedback)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    minimums = torch.tensor([-10.0, -10.0], dtype = torch.double, device = device)
    maximums = torch.tensor([10.0, 10.0], dtype = torch.double, device = device)
    
    starting_points = 1
    
    a = rescale(torch.rand(starting_points, 2, dtype = torch.double, device = device), minimums, maximums)
    reference = torch.tensor([0.00, 0.00], dtype = torch.double, device = device)
    
    weights = generate_weights(2).to(torch.double).to(device = device)
    
    iters = 50
    
    utilityA = torch.rand(iters, dtype = torch.double, device = device)
    utilityB = 1 - utilityA
    utilitiesX = torch.stack((utilityA, utilityB), dim = 1)
    
    q = 8
    
    for i in tqdm(range(iters)):
        f_a = normalize(-objective2(a.T), 0, 0.206261) # 0.206261 is the global maximum, 0 is the global minimum
        f_a = f_a.view(len(f_a), 1)
        f_a2 = normalize(-objective3(a.T), 0, 19.2085) # 19.2085 is the global maximum, 0 is the global minimum
        f_a2 = f_a2.view(len(f_a2), 1)
        
        utilities = utilitiesX[i]
        
        if (acquisition_function == "qEI"):
            optimizer = ExpectedImprovementCalculator(a, f_a, f_a2, torch.max(chebyshev_scalar(f_a, f_a2, reference, utilities)), minimums, 
                                                 maximums, reference, cost_aware = False, q = q, weights = weights, utilities = utilities, device = device, seed = i, normalization = 10)
        else:
            optimizer = UpperConfidenceBoundCalculator(a, f_a, f_a2, torch.max(chebyshev_scalar(f_a, f_a2, reference, utilities)), torch.tensor([-10.0, -10.0]), 
                                                       torch.tensor([10.0, 10.0]), reference, cost_aware = True, weights = weights, utilities = utilities, device = device, seed = i, normalization = 10, beta = 3)
        next_a = optimizer.getNext(i+1)
        a = torch.cat([a, next_a], dim = 0)
        
    if (acquisition_function == "qEI"):
        torch.save(a[starting_points:], "queried_data_normal_qei.pt")
    else:
        torch.save(a[starting_points:], "queried_data_normal_ucb.pt")
    
    final_utilities = torch.cat((f_a, f_a2), dim = 1)
    pareto_points = final_utilities[is_non_dominated(final_utilities)]
    print(pareto_points)
    
    hypervolume = Hypervolume(torch.tensor([0.00, 0.00], dtype = torch.double, device = device))
    dominated_hypervolume = hypervolume.compute(pareto_points)
    print(dominated_hypervolume)

def misspecifiedTestQEI():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    minimums = torch.tensor([-10.0, -10.0], dtype = torch.double, device = device)
    maximums = torch.tensor([10.0, 10.0], dtype = torch.double, device = device)
    
    starting_points = 1
    iters = 50
    
    a = rescale(torch.rand(starting_points, 2, dtype = torch.double, device = device), minimums, maximums)
    reference = torch.tensor([0.00, 0.00], dtype = torch.double, device = device)
    
    weights = generate_weights(2).to(torch.double).to(device = device)

    utilityA = torch.rand(iters, dtype = torch.double, device = device)
    utilityB = 1 - utilityA
    utilitiesX = torch.stack((utilityA, utilityB), dim = 1)
    
    q = 16
    delay = 25
    
    buffer = torch.empty((iters * q, 2), dtype = torch.double)
    estimate_f1 = None
    estimate_f2 = None
    optimizer = None
    cost_aware = False
    
    for i in tqdm(range(iters)):
        if (i % delay == 0 and i > 0):
            # release the evaluations
            a = torch.cat([a, buffer[i-delay:i]], dim = 0)
        
        if (i % q != 0):
            # still have points to query
            continue
        
        if (i % delay != 0):
            filler = buffer[len(a) - 1:i]
            estimate_f1, estimate_f2 = optimizer.predict(filler)
        
        f_a = normalize(objective4(a.T), 0, 1.0) # 1.0 is the global maximum, 0 is the minimum
        f_a = f_a.view(len(f_a), 1)
        f_a2 = normalize(objective5(a.T), 0, 80.708) # 80.708 is the global maximum, 0 is the minimum
        f_a2 = f_a2.view(len(f_a2), 1)
        
        
        utilities = utilitiesX[i:i+q]
        
        if (estimate_f1 == None):
            x = torch.transpose(torch.stack((f_a, f_a2), dim = 1).repeat(1, 1, min(q, iters - i)), 1, 2)
            optimizer = ExpectedImprovementCalculator(a, f_a, f_a2, torch.max(chebyshev_scalar_batch(x, reference, utilities)), 
                                                    minimums, maximums, reference, cost_aware = cost_aware, q = q, weights = weights, 
                                                    utilities = utilities, device = device, seed = i, normalization = 5)
        else:
            fa = torch.cat([f_a, estimate_f1], dim = 0)
            fa2 = torch.cat([f_a2, estimate_f2], dim = 0)
            x = torch.transpose(torch.stack((fa, fa2), dim = 1).repeat(1, 1, min(q, iters - i)), 1, 2)
            optimizer = ExpectedImprovementCalculator(torch.cat([a, filler], dim = 0), fa, fa2, 
                                                    torch.max(chebyshev_scalar_batch(x, reference, utilities)),
                                                    minimums, maximums, reference, cost_aware = cost_aware, q = q, weights = weights, 
                                                    utilities = utilities, device = device, seed = i, normalization = 5)
        buffer[i:i+min(q, iters - i)] = optimizer.getNext(i+1, min(q, iters - i))
        estimate_f1 = None
        
        
    a = torch.cat([a, buffer[iters-delay:iters]], dim = 0)
    
    f_a = normalize(objective4(a.T), 0, 1.0)
    f_a = f_a.view(len(f_a), 1)
    f_a2 = normalize(objective5(a.T), 0, 80.708)
    f_a2 = f_a2.view(len(f_a2), 1)
    
    torch.save(a[starting_points:], "queried_data_delayed_qei.pt")
    
    final_utilities = torch.cat((f_a, f_a2), dim = 1)
    pareto_points = final_utilities[is_non_dominated(final_utilities)]
    print(pareto_points)
    
    hypervolume = Hypervolume(torch.tensor([0.00, 0.00], dtype = torch.double, device = device))
    dominated_hypervolume = hypervolume.compute(pareto_points)
    print(dominated_hypervolume)
    
def misspecifiedTestUCB():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    minimums = torch.tensor([-5.0, -5.0], dtype = torch.double, device = device)
    maximums = torch.tensor([5.0, 5.0], dtype = torch.double, device = device)
    
    starting_points = 1
    iters = 50
    
    a = rescale(torch.rand(starting_points, 2, dtype = torch.double, device = device), minimums, maximums)
    #a = rescale(torch.rand(8, 2, dtype = torch.double), torch.tensor([0.0, 0.0]), torch.tensor([10.0, 10.0]))
    reference = torch.tensor([0.00, 0.00], dtype = torch.double, device = device)
    
    weights = generate_weights(2).to(torch.double).to(device = device)
     
    utilityA = torch.rand(iters, dtype = torch.double, device = device)
    utilityB = 1 - utilityA
    utilitiesX = torch.stack((utilityA, utilityB), dim = 1)
    delay = 5
    
    buffer = torch.empty((iters, 2), dtype = torch.double)
    estimate_f1 = None
    estimate_f2 = None
    optimizer = None
    cost_aware = True
    f_a = None
    estimated_values1, estimated_values2 = (None, None)
    
    for i in tqdm(range(iters)):
        if (i % delay == 0 and i > 0):
            # release the evaluations
            a = torch.cat([a, buffer[i-delay:i]], dim = 0)
        
        if (i % delay != 0):
            filler = buffer[len(a) - 1:i]
            estimate_f1, estimate_f2 = optimizer.predict(buffer[i-1].view(1, -1))
            if (estimated_values1 == None):
                estimated_values1 = estimate_f1.detach()
                estimated_values2 = estimate_f2.detach()
            else:
                estimated_values1 = torch.cat([estimated_values1, estimate_f1], dim = 0)
                estimated_values2 = torch.cat([estimated_values2, estimate_f2], dim = 0)
        
        f_a = normalize(objective4(a.T), 0, 1.0) # 1.0 is the global maximum, 0 is the minimum
        f_a = f_a.view(len(f_a), 1)
        f_a2 = normalize(objective5(a.T), 0, 80.708) # 80.708 is the global maximum, 0 is the minimum
        f_a2 = f_a2.view(len(f_a2), 1)
        
        
        utilities = utilitiesX[i]
        
        if (estimate_f1 == None):
            estimated_values1, estimated_values2 = (None, None)
            optimizer = UpperConfidenceBoundCalculator(a, f_a, f_a2, torch.max(chebyshev_scalar(f_a, f_a2, reference, utilities)), minimums, 
                                                      maximums, reference, cost_aware = cost_aware, weights = weights, utilities = utilities, device = device, seed = i, normalization = 5, beta = 1.2)
        else:
            fa = torch.cat([f_a, estimated_values1], dim = 0)
            fa2 = torch.cat([f_a2, estimated_values2], dim = 0)
            optimizer = UpperConfidenceBoundCalculator(torch.cat([a, filler], dim = 0), fa, fa2, torch.max(chebyshev_scalar(fa, fa2, reference, utilities)), minimums, 
                                                      maximums, reference, cost_aware = cost_aware, weights = weights, utilities = utilities, device = device, seed = i, normalization = 5, beta = 1.2)
        buffer[i] = optimizer.getNext(i+1)
        estimate_f1 = None
        
        
    a = torch.cat([a, buffer[iters-delay:iters]], dim = 0)
    
    f_a = normalize(objective4(a.T), 0, 1.0)
    f_a = f_a.view(len(f_a), 1)
    f_a2 = normalize(objective5(a.T), 0, 80.708)
    f_a2 = f_a2.view(len(f_a2), 1)
    
    torch.save(a[starting_points:], "queried_data_delayed_ucb.pt")
    
    final_utilities = torch.cat((f_a, f_a2), dim = 1)
    pareto_points = final_utilities[is_non_dominated(final_utilities)]
    print(pareto_points)
    
    hypervolume = Hypervolume(torch.tensor([0.00, 0.00], dtype = torch.double, device = device))
    dominated_hypervolume = hypervolume.compute(pareto_points)
    print(dominated_hypervolume)
    

def visualizeDelayed():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    minimums = torch.tensor([-10.0, -10.0], dtype = torch.double, device = device) / 2
    maximums = torch.tensor([10.0, 10.0], dtype = torch.double, device = device) / 2
    
    qei_unaware85 = rescale(torch.rand(50, 2, dtype = torch.double, device = device), minimums, maximums)
    
    # results = [qei_unaware85, qei_unaware1610, qei_unaware1625]
    
    results = [qei_unaware85]
    
    _, ax = plt.subplots()
    
    volumes = []
    i = 0
    
    for res in results:
        volumes.append([])
        for t in range(50):
            
            f_a = normalize(objective4(res[:t].T), 0, 1.0)
            f_a = f_a.view(len(f_a), 1)
            f_a2 = normalize(objective5(res[:t].T), 0, 80.708)
            f_a2 = f_a2.view(len(f_a2), 1)
        
            final_utilities = torch.cat((f_a, f_a2), dim = 1)
            pareto_points = final_utilities[is_non_dominated(final_utilities)]
        
            hypervolume = Hypervolume(torch.tensor([0.00, 0.00], dtype = torch.double))
            dominated_hypervolume = hypervolume.compute(pareto_points)
            print(dominated_hypervolume)
            volumes[i].append(dominated_hypervolume)
        i += 1
    
    
    #plt.rc('legend', fontsize=13, loc='upper left')
    ax.set_xlabel("Timestep t", fontsize=16)
    ax.set_ylabel(r"Dominated hypervolume", fontsize=16)
    #ax.plot(np.arange(start = 1, stop = 51, step = 1), volumes[0], label = r"d = 5")
    #ax.plot(np.arange(start = 1, stop = 51, step = 1), volumes[1], label = r"d = 10")
    #ax.plot(np.arange(start = 1, stop = 51, step = 1), volumes[2], label = r"d = 25")
    ax.plot(np.arange(start = 1, stop = 51, step = 1), volumes[0])
    plt.locator_params(axis='both', nbins=6)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    #plt.legend()
    plt.show()

def visualize2d():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    minimums = torch.tensor([-10.0, -10.0], dtype = torch.double, device = device)
    maximums = torch.tensor([10.0, 10.0], dtype = torch.double, device = device)
    
    qei_unaware = torch.load('qei_unaware50.pt')
    ucb_aware = torch.load("ucb_aware50.pt")
    qei_aware = torch.load('qei_aware50.pt')
    
    
    results = [ucb_aware, qei_aware, qei_unaware]
    
    _, ax = plt.subplots()
    
    for res in results:
        
        f_a = normalize(-objective2(res.T), 0, 0.206261)
        f_a = f_a.view(len(f_a), 1)
        f_a2 = normalize(-objective3(res.T), 0, 19.2085)
        f_a2 = f_a2.view(len(f_a2), 1)
        
        final_utilities = torch.cat((f_a, f_a2), dim = 1)
        pareto_points = final_utilities[is_non_dominated(final_utilities)]
        # print(pareto_points)
        
        hypervolume = Hypervolume(torch.tensor([0.00, 0.00], dtype = torch.double))
        dominated_hypervolume = hypervolume.compute(pareto_points)
        print(dominated_hypervolume)
        
        pareto_points_np = pareto_points.T.detach().numpy()
        
        ax.scatter(pareto_points_np[0], pareto_points_np[1], label = "Pareto frontier")
        break
    
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # Set the legend font size
    plt.rc('legend', fontsize=15, loc='upper left')
    plt.xlabel("Utility of f1", fontsize=16)
    plt.ylabel("Utility of f2", fontsize=16)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    ax.grid(True)
    plt.show()
    
    _, ax = plt.subplots() 
    
    
    #res = torch.reshape(results[2], (-1, 8, 2))
    #costs = torch.sum(abs(res), axis = 1) / 8
    # costs = torch.cumsum(costs, dim = 0)
    costs_0 = torch.cumsum(abs(results[0]), dim = 0)
    #costs_1 = torch.cumsum(abs(results[1]), dim = 0)
    #costs_2 = torch.cumsum(abs(results[2]), dim = 0)
    print(costs_0[-1][1] - costs_0[-1][0])
    #print(costs_1[-1][1] - costs_1[-1][0])
    #print(costs_2[-1][1] - costs_2[-1][0])
    #plt.rc('legend', fontsize=15, loc='upper left')
    ax.set_xlabel("Timestep t", fontsize=16)
    ax.set_ylabel(r"Accumulation of $x_{2} - x_{1}$", fontsize=16)
    #plt.ylim([-5, 100])
    ax.plot(np.arange(start = 1, stop = 51, step = 1), costs_0.T.detach().numpy()[1] - costs_0.T.detach().numpy()[0])
    #ax.plot(np.arange(start = 1, stop = 51, step = 1), costs_1.T.detach().numpy()[1] - costs_1.T.detach().numpy()[0], label = r"d = 10")
    #ax.plot(np.arange(start = 1, stop = 51, step = 1), costs_2.T.detach().numpy()[1] - costs_2.T.detach().numpy()[0], label = r"d = 25")
    plt.locator_params(axis='both', nbins=6)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    #plt.legend()
    plt.show()
    
    
if __name__ == '__main__':
    sd = 101
    
    torch.manual_seed(sd)
    seed(sd)
    np.random.seed(sd)
    
    test2d()

