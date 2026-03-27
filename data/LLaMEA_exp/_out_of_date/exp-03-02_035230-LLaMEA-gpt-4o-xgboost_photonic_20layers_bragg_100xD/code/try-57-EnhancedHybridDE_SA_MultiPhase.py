import numpy as np

class EnhancedHybridDE_SA_MultiPhase:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.F = 0.5  # Initial Differential Evolution scaling factor
        self.CR = 0.9  # Crossover probability
        self.alpha = 0.9  # Cooling rate for Simulated Annealing
        self.beta = 0.99  # Adaptive factor for DE parameters
        self.eexplore_weight = 0.1  # Weight for exploration-exploitation balance
        self.dynamic_resize_factor = dim // 2  # Dynamic resizing factor

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
            for phase in range(3):  # Multi-phase adaptation
                if phase == 1:  # Exploration phase
                    self.F = 0.9
                    self.CR = 0.8
                elif phase == 2:  # Exploitation phase
                    self.F = 0.3
                    self.CR = 0.6
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
                T *= self.alpha
                
                # Dynamic exploration-exploitation adjustment with Lévy flights
                global_best_idx = np.argmin(fitness)
                global_best = population[global_best_idx]
                for j in range(self.population_size):
                    if np.random.rand() < 0.1:  # 10% chance to refine based on global best
                        distance = np.linalg.norm(population[j] - global_best)
                        adjust_factor = np.exp(-self.eexplore_weight * distance)
                        population[j] = population[j] + adjust_factor * (global_best - population[j]) + self.levy_flight(self.dim)
                        population[j] = np.clip(population[j], bounds[:, 0], bounds[:, 1])
                        fitness[j] = func(population[j])
                        eval_budget += 1
                        if eval_budget >= self.budget:
                            break

            # Adaptive population resizing for diverse search phases
            if phase == 0:
                self.population_size = max(self.initial_population_size // 2, 5)
            elif phase == 1:
                self.population_size = min(self.initial_population_size + self.dynamic_resize_factor, self.budget - eval_budget)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]