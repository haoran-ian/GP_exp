import numpy as np
from numpy.random import rand, randn

class HybridDE_SA_CMA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Initial Differential Evolution scaling factor
        self.CR = 0.9  # Crossover probability
        self.alpha = 0.9  # Cooling rate for Simulated Annealing
        self.beta = 0.99  # Adaptive factor for DE parameters
        self.cma_lambda = 4 + int(3 * np.log(dim))  # Population size
        self.cma_mu = self.cma_lambda // 2
        self.weights = np.log(self.cma_mu + 0.5) - np.log(np.arange(1, self.cma_mu + 1))
        self.weights /= np.sum(self.weights)
        self.sigma = 0.3  # Initial step-size
        self.cov = np.eye(dim)
    
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_budget = self.population_size
        T = 1.0  # Initial temperature for Simulated Annealing
        cma_centroid = np.mean(population, axis=0)
        
        while eval_budget < self.budget:
            # Differential Evolution and Simulated Annealing
            for i in range(self.population_size):
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = func(trial)
                if eval_budget >= self.budget:
                    break
                eval_budget += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / T)
                    if np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness
            
            # Covariance Matrix Adaptation
            if eval_budget < self.budget:
                z = randn(self.cma_lambda, self.dim)
                y = z @ np.linalg.cholesky(self.cov)
                cma_offspring = cma_centroid + self.sigma * y
                cma_offspring = np.clip(cma_offspring, bounds[:, 0], bounds[:, 1])
                cma_fitness = np.array([func(ind) for ind in cma_offspring])
                eval_budget += self.cma_lambda
                best_indices = np.argsort(cma_fitness)[:self.cma_mu]
                cma_centroid = np.dot(self.weights, cma_offspring[best_indices])
                z_best = z[best_indices]
                self.cov = (1 - 1/self.cma_mu) * self.cov + (1/self.cma_mu) * z_best.T @ np.diag(self.weights) @ z_best
            
            # Cooling schedule for Simulated Annealing
            T *= self.alpha
            
            # Adaptive parameter control
            if np.random.rand() < 0.1:  # 10% chance to adapt parameters
                self.F = self.F * self.beta + 0.1 * np.random.rand()  # Randomly tweak F
                self.CR = self.CR * self.beta + 0.1 * np.random.rand()  # Randomly tweak CR
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]