import numpy as np

class EnhancedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.max_iterations = budget // self.population_size
        self.temperature = 100.0
        self.cooling_rate = 0.95  # Slightly faster cooling
        self.mutation_scaling = 0.5  # Adaptive mutation scaling

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iterations):
            adaptive_rate = 0.9 - 0.5 * (iteration / self.max_iterations)  # Dynamically adjusted crossover rate
            
            for i in range(self.population_size):
                indices = np.random.permutation(self.population_size)
                x1, x2, x3 = population[indices[:3]]

                # Differential Evolution Mutation with adaptive scaling
                F = self.mutation_scaling + 0.1 * np.random.rand()  # Stochastic and adaptive scaling
                mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Multi-Crossover
                crossover_mask = np.random.rand(self.dim) < adaptive_rate
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Additional stochastic crossover
                alt_crossover_mask = np.random.rand(self.dim) < (1 - adaptive_rate)
                alt_trial = np.where(alt_crossover_mask, mutant, population[i])

                # Evaluate both trials and select the better one
                trial_fitness = func(trial)
                alt_trial_fitness = func(alt_trial)
                if trial_fitness < alt_trial_fitness:
                    selected_trial, selected_fitness = trial, trial_fitness
                else:
                    selected_trial, selected_fitness = alt_trial, alt_trial_fitness

                # Simulated Annealing Acceptance
                if selected_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - selected_fitness) / self.temperature):
                    population[i] = selected_trial
                    fitness[i] = selected_fitness

                # Update best solution found
                if selected_fitness < best_fitness:
                    best_solution = selected_trial
                    best_fitness = selected_fitness

            # Cooling
            self.temperature *= self.cooling_rate

        return best_solution