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
        self.elitism_rate = 0.1  # Percentage of elite individuals to retain

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        temperature = self.initial_temperature

        for iteration in range(self.max_iterations):
            adaptive_rate = 0.9 - 0.5 * (iteration / self.max_iterations)  # Dynamically adjusted crossover rate
            num_elites = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(fitness)[:num_elites]
            elites = population[elite_indices]

            for i in range(self.population_size):
                if i not in elite_indices:
                    indices = np.random.permutation(self.population_size)
                    x1, x2, x3 = population[indices[:3]]

                    # Differential Evolution Mutation
                    mutant = x1 + 0.8 * (x2 - x3)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                    # Crossover
                    crossover_mask = np.random.rand(self.dim) < adaptive_rate
                    trial = np.where(crossover_mask, mutant, population[i])
                    
                    # Diversity enhancement by random perturbation
                    if np.random.rand() < 0.1:
                        trial += np.random.normal(0, 0.1, self.dim)

                    # Simulated Annealing Acceptance
                    trial_fitness = func(trial)
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / temperature)
                    if trial_fitness < fitness[i] or np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness

                    # Update best solution found
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

            # Ensure elitism by maintaining elite individuals
            population[:num_elites] = elites

            # Cooling
            temperature *= self.cooling_rate

        return best_solution