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
        self.cooling_rate = 0.99
        self.elitism_rate = 0.1  # Rate for elitism

    def __call__(self, func):
        # Initialize population and evaluate fitness
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iterations):
            adaptive_rate = 0.9 - 0.5 * (iteration / self.max_iterations)  # Dynamically adjusted crossover rate
            elite_count = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            new_population = population[elite_indices]  # Keep elite solutions

            while len(new_population) < self.population_size:
                indices = np.random.permutation(self.population_size)
                x1, x2, x3 = population[indices[:3]]

                # Differential Evolution Mutation
                mutant = x1 + np.random.uniform(0.5, 1.0) * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < adaptive_rate
                trial = np.where(crossover_mask, mutant, population[indices[0]])

                # Simulated Annealing Acceptance
                trial_fitness = func(trial)
                if trial_fitness < fitness[indices[0]] or np.random.rand() < np.exp((fitness[indices[0]] - trial_fitness) / self.temperature):
                    new_population = np.vstack([new_population, trial])
                
                # If budget allows, evaluate trial and consider for best solution
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

            population = new_population
            fitness = np.array([func(ind) for ind in population])

            # Cooling
            self.temperature *= self.cooling_rate

        return best_solution