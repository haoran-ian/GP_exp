import numpy as np

class EnhancedAdaptiveLevyDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Initial Differential Evolution scaling factor
        self.CR = 0.9  # Crossover probability
        self.alpha = 0.9  # Cooling rate for Simulated Annealing
        self.beta = 0.99  # Adaptive factor for DE parameters
        self.eexplore_weight = 0.1  # Weight for exploration-exploitation balance

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
        T = 1.0  # Initial temperature for Simulated Annealing
        
        while eval_budget < self.budget:
            for i in range(self.population_size):
                # Differential Evolution mutation and crossover
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])
                
                # Simulated Annealing acceptance criterion
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

            # Cooling schedule for Simulated Annealing
            T *= self.alpha * 0.95  # Slightly increase cooling rate
            
            # Adaptive parameter control with enhancement
            self.F = self.F * self.beta + self.eexplore_weight * np.random.rand() * 0.1  # Fine-tune F
            self.CR = self.CR * (self.beta + 0.01) + self.eexplore_weight * np.random.rand() * 0.1  # Fine-tune CR
            
            # Dynamic exploration-exploitation adjustment with adaptive Lévy flights
            global_best_idx = np.argmin(fitness)
            global_best = population[global_best_idx]
            for j in range(self.population_size):
                if np.random.rand() < 0.2:  # 20% chance to refine based on global best
                    distance = np.linalg.norm(population[j] - global_best)
                    adjust_factor = np.exp(-self.eexplore_weight * distance)
                    levy_step = self.levy_flight(self.dim) * adjust_factor
                    population[j] = population[j] + adjust_factor * (global_best - population[j]) + levy_step
                    population[j] = np.clip(population[j], bounds[:, 0], bounds[:, 1])
                    fitness[j] = func(population[j])
                    eval_budget += 1
                    if eval_budget >= self.budget:
                        break
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]