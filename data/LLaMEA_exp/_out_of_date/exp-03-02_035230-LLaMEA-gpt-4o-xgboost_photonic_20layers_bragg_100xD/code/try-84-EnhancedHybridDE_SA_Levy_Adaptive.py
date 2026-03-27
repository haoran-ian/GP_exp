import numpy as np

class EnhancedHybridDE_SA_Levy_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.F = 0.5
        self.CR = 0.9
        self.alpha = 0.9
        self.beta = 0.99
        self.eexplore_weight = 0.1
    
    def levy_flight(self, size, scale=1.0):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = scale * u / np.abs(v)**(1 / beta)
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
                trial = np.where(np.random.rand(self.dim) < T, mutant, population[i])
                
                trial_fitness = func(trial)
                if eval_budget >= self.budget:
                    break
                eval_budget += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / (T + 1e-10))
                    if np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness

            T *= self.alpha * (1 - eval_budget / self.budget)
            if np.random.rand() < 0.2:
                self.F = np.std(fitness) / np.mean(fitness)  # Adaptive mutation scaling
                self.CR = self.CR * (self.beta + np.random.rand() * 0.05)
            
            global_best_idx = np.argmin(fitness)
            global_best = population[global_best_idx]
            for j in range(self.population_size):
                if np.random.rand() < 0.1:
                    distance = np.linalg.norm(population[j] - global_best)
                    scale_factor = 1.0 if distance > 0.1 else 0.5
                    adjust_factor = np.exp(-self.eexplore_weight * distance)
                    population[j] = population[j] + adjust_factor * (global_best - population[j]) + self.levy_flight(self.dim, scale=scale_factor)
                    population[j] = np.clip(population[j], bounds[:, 0], bounds[:, 1])
                    fitness[j] = func(population[j])
                    eval_budget += 1
                    if eval_budget >= self.budget:
                        break
            
            if np.random.rand() < 0.05:
                improvement = np.max(fitness) - np.min(fitness)
                crowding_distance = np.zeros(self.population_size)
                for m in range(self.dim):
                    sorted_idx = np.argsort(population[:, m])
                    crowding_distance[sorted_idx[0]] = np.inf
                    crowding_distance[sorted_idx[-1]] = np.inf
                    for n in range(1, self.population_size - 1):
                        crowding_distance[sorted_idx[n]] += (population[sorted_idx[n + 1], m] - population[sorted_idx[n - 1], m])
                if improvement < 1e-6:
                    self.population_size = max(int(self.population_size * 0.9), 5)
                else:
                    self.population_size = min(int(self.population_size * 1.1), self.initial_population_size)
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]