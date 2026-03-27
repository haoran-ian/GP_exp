import numpy as np

class DualPopulationMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Refined cooling rate
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.elite_fraction = 0.2

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        elite_size = int(self.elite_fraction * self.population_size)
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            # Sort population by fitness
            sorted_indices = np.argsort(fitness)
            elite_population = population[sorted_indices[:elite_size]]
            general_population = population[sorted_indices[elite_size:]]

            # Differential Evolution on general population
            for i in range(self.population_size - elite_size):
                idxs = [idx for idx in range(self.population_size - elite_size) if idx != i]
                a, b, c = general_population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.05 * np.cos(budget_used / self.budget * np.pi)
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                trial = np.where(crossover, mutant, general_population[i])

                # Simulated Annealing: accept based on Metropolis criterion
                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[sorted_indices[elite_size + i]] or \
                        np.random.rand() < np.exp((fitness[sorted_indices[elite_size + i]] - trial_fitness) / self.temperature):
                    general_population[i] = trial
                    fitness[sorted_indices[elite_size + i]] = trial_fitness

                if budget_used >= self.budget:
                    break

            # Explore with elite population
            for i in range(elite_size):
                small_mutation = np.random.normal(0, 0.1, self.dim)
                explorer = np.clip(elite_population[i] + small_mutation, lb, ub)
                explorer_fitness = func(explorer)
                budget_used += 1
                if explorer_fitness < fitness[sorted_indices[i]]:
                    elite_population[i] = explorer
                    fitness[sorted_indices[i]] = explorer_fitness

                if budget_used >= self.budget:
                    break

            # Merge populations back
            population = np.vstack((elite_population, general_population))

            # Cool down temperature more strategically
            self.temperature *= self.cooling_rate - 0.01 * np.sqrt(budget_used / self.budget)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]