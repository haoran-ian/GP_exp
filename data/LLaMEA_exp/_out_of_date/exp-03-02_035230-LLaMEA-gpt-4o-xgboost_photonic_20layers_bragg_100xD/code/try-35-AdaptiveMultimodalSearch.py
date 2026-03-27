import numpy as np

class AdaptiveMultimodalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 15 * dim
        self.F = 0.5
        self.CR = 0.9
        self.alpha = 0.95
        self.beta_factor = 0.99
        self.explore_weight = 0.1

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v)**(1 / beta)
        return step

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_budget = self.population_size
        T = 1.0
        
        while eval_budget < self.budget:
            for i in range(self.population_size):
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = func(trial)
                if eval_budget >= self.budget:
                    break
                eval_budget += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / T):
                    population[i] = trial
                    fitness[i] = trial_fitness

            T *= self.alpha
            
            self.F = np.clip(self.F * self.beta_factor + self.explore_weight * np.random.rand(), 0.4, 0.9)
            self.CR = np.clip(self.CR * (self.beta_factor + 0.01) + self.explore_weight * np.random.rand(), 0.8, 1.0)
            
            global_best_idx = np.argmin(fitness)
            global_best = population[global_best_idx]
            for j in range(self.population_size):
                if np.random.rand() < 0.15:
                    distance = np.linalg.norm(population[j] - global_best)
                    adjust_factor = np.exp(-self.explore_weight * distance)
                    population[j] = population[j] + adjust_factor * (global_best - population[j]) + self.levy_flight(self.dim)
                    population[j] = np.clip(population[j], bounds[:, 0], bounds[:, 1])
                    fitness[j] = func(population[j])
                    eval_budget += 1
                    if eval_budget >= self.budget:
                        break
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]