import numpy as np

class EnhancedHybridDE_SA_Levy_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.F = 0.5  # Initial Differential Evolution scaling factor
        self.CR = 0.9  # Crossover probability
        self.alpha = 0.9  # Cooling rate for Simulated Annealing
        self.beta = 0.99  # Adaptive factor for DE parameters
        self.eexplore_weight = 0.1  # Exploration-exploitation balance weight
        self.learning_rate = 0.1  # Adaptive learning rate
        self.levy_scale_factor = 1.0  # Initial Lévy flight scale factor

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
        T = 1.0  # Initial temperature for Simulated Annealing

        while eval_budget < self.budget:
            for i in range(self.population_size):
                # Differential Evolution mutation and crossover
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(np.random.rand(self.dim) < T, mutant, population[i])  # Dynamic crossover operator

                # Simulated Annealing acceptance criterion
                trial_fitness = func(trial)
                if eval_budget >= self.budget:
                    break
                eval_budget += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / (T + 1e-10))  # Adjusted selection pressure
                    if np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness

            # Cooling schedule for Simulated Annealing
            T *= self.alpha
            
            # Adaptive parameter control
            if np.random.rand() < 0.2:  # 20% chance to adapt parameters
                self.F = np.random.rand() * self.beta + 0.4  # Self-adaptive scaling factor
                self.CR = self.CR * (self.beta + np.random.rand() * 0.05)  # Adjust CR with random perturbation
            
            # Dynamic exploration-exploitation adjustment with Lévy flights
            global_best_idx = np.argmin(fitness)
            global_best = population[global_best_idx]
            for j in range(self.population_size):
                if np.random.rand() < 0.1:  # 10% chance to refine based on global best
                    distance = np.linalg.norm(population[j] - global_best)
                    scale_factor = self.levy_scale_factor * (1 + np.random.rand() * 0.5)  # Varying Lévy flight scale
                    adjust_factor = np.exp(-self.eexplore_weight * distance)
                    population[j] = population[j] + adjust_factor * (global_best - population[j]) + self.levy_flight(self.dim, scale=scale_factor)
                    population[j] = np.clip(population[j], bounds[:, 0], bounds[:, 1])
                    fitness[j] = func(population[j])
                    eval_budget += 1
                    if eval_budget >= self.budget:
                        break
            
            # Multistage adaptive learning rate adjustment
            if eval_budget / self.budget > 0.5 and eval_budget / self.budget < 0.75:
                self.learning_rate *= 1.1  # Increase learning rate
            elif eval_budget / self.budget >= 0.75:
                self.learning_rate *= 0.9  # Decrease learning rate
                
            # Dynamic population resizing based on progress
            if np.random.rand() < 0.05:  # Occasionally adjust population size
                improvement = np.max(fitness) - np.min(fitness)
                if improvement < 1e-6:  # If minimal progress, reduce population
                    self.population_size = max(int(self.population_size * 0.9), 5)
                else:  # Otherwise, maintain or slightly increase
                    self.population_size = min(int(self.population_size * 1.1), self.initial_population_size)
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]