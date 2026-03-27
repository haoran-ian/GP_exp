import numpy as np

class EnhancedHybridDE_SA_Levy_Chaos:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.alpha = 0.95  # Slightly increase cooling rate
        self.beta = 0.99
        self.eexplore_weight = 0.1
        self.chaos_r = 3.99  # Parameter for chaotic map

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v)**(1 / beta)
        return step

    def chaotic_map(self, x):
        return self.chaos_r * x * (1 - x)  # Logistic map

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_budget = self.population_size
        T = 1.0
        chaotic_var = np.random.rand()

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
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / T)
                    if np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness

            T *= self.alpha

            if np.random.rand() < 0.3:
                self.F = self.F * self.beta + self.eexplore_weight * chaotic_var
                chaotic_var = self.chaotic_map(chaotic_var)
                self.CR = self.CR * (self.beta + chaotic_var * 0.05)
            
            global_best_idx = np.argmin(fitness)
            global_best = population[global_best_idx]
            for j in range(self.population_size):
                if np.random.rand() < 0.1:
                    distance = np.linalg.norm(population[j] - global_best)
                    adjust_factor = np.exp(-self.eexplore_weight * distance)
                    population[j] = population[j] + adjust_factor * (global_best - population[j]) + self.levy_flight(self.dim)
                    population[j] = np.clip(population[j], bounds[:, 0], bounds[:, 1])
                    fitness[j] = func(population[j])
                    eval_budget += 1
                    if eval_budget >= self.budget:
                        break
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]