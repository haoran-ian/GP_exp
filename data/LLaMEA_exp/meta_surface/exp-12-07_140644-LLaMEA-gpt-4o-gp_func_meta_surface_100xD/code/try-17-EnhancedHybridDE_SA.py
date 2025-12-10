import numpy as np

class EnhancedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.max_iterations = budget // self.population_size
        self.initial_temperature = 100.0
        self.cooling_rate = 0.99
        
        # Adaptive parameters
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.temperature = self.initial_temperature

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iterations):
            for i in range(self.population_size):
                indices = np.random.permutation(self.population_size)
                x1, x2, x3 = population[indices[:3]]

                # Differential Evolution Mutation with Adaptive Control
                mutant = x1 + self.mutation_factor * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover with Adaptive Control
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover_mask, mutant, population[i])

                # Simulated Annealing Acceptance with Elitism
                trial_fitness = func(trial)
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Update best solution found
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness
                    
            # Cooling
            self.temperature *= self.cooling_rate
            
            # Adaptive control of parameters
            self.mutation_factor = 0.5 + 0.3 * np.cos(iteration * np.pi / self.max_iterations)
            self.crossover_rate = 0.7 + 0.2 * np.sin(iteration * np.pi / self.max_iterations)

        return best_solution