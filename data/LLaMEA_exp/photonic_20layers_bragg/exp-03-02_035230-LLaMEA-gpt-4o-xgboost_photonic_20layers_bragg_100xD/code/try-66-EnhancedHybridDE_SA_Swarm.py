import numpy as np

class EnhancedHybridDE_SA_Swarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.F = 0.5  # Initial DE scaling factor
        self.CR = 0.9  # Crossover probability
        self.alpha = 0.95  # Cooling rate for SA
        self.beta = 0.97  # Adaptive factor for DE parameters
        self.eexplore_weight = 0.15  # Exploration-exploitation balance weight
    
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
        T = 1.0  # Initial temperature for SA
        
        while eval_budget < self.budget:
            global_best_idx = np.argmin(fitness)
            global_best = population[global_best_idx]
            
            for i in range(self.population_size):
                # Swarm-based differential evolution
                if np.random.rand() < 0.3:  # 30% chance to use Swarm Intelligence
                    a, b = population[np.random.choice(self.population_size, 2, replace=False)]
                    mutant = np.clip(a + self.F * (b - global_best), bounds[:, 0], bounds[:, 1])
                else:
                    a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                    mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
                
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(np.random.rand(self.dim) < T, mutant, population[i])
                
                # SA acceptance criterion
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

            if np.random.rand() < 0.25:
                self.F = np.random.rand() * self.beta + 0.3
                self.CR = self.CR * (self.beta + np.random.rand() * 0.1)

            for j in range(self.population_size):
                if np.random.rand() < 0.15:
                    distance = np.linalg.norm(population[j] - global_best)
                    scale_factor = 0.9 if distance > 0.1 else 0.3
                    adjust_factor = np.exp(-self.eexplore_weight * distance)
                    population[j] = population[j] + adjust_factor * (global_best - population[j]) + self.levy_flight(self.dim, scale=scale_factor)
                    population[j] = np.clip(population[j], bounds[:, 0], bounds[:, 1])
                    fitness[j] = func(population[j])
                    eval_budget += 1
                    if eval_budget >= self.budget:
                        break
            
            if np.random.rand() < 0.1:
                improvement = np.max(fitness) - np.min(fitness)
                if improvement < 1e-5:
                    self.population_size = max(int(self.population_size * 0.85), 5)
                else:
                    self.population_size = min(int(self.population_size * 1.15), self.initial_population_size)
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]