import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_size = 10 * dim
        self.temperature = 1.0
        self.cooling_rate = 0.99
        self.scale_factor = 0.8
        self.crossover_rate = 0.9

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        population_size = self.base_population_size

        # Initialize population
        population = np.random.uniform(lower_bound, upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while evals < self.budget:
            new_population = []
            for i in range(population_size):
                # Adaptive Differential Evolution mutation and crossover
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.scale_factor * (b - c), lower_bound, upper_bound)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                offspring = np.where(cross_points, mutant, population[i])

                # Evaluate offspring
                offspring_fitness = func(offspring)
                evals += 1

                # Replacement strategy
                if offspring_fitness < fitness[i]:
                    new_population.append(offspring)
                    fitness[i] = offspring_fitness
                    # Simulated annealing acceptance criterion
                    if offspring_fitness < best_fitness or np.exp((best_fitness - offspring_fitness) / self.temperature) > np.random.rand():
                        best_solution = offspring
                        best_fitness = offspring_fitness
                else:
                    new_population.append(population[i])

                # Adjust population size dynamically
                if evals % (0.1 * self.budget) == 0:
                    population_size = min(int(1.1 * population_size), self.budget - evals + population_size)

                # Update temperature
                self.temperature *= self.cooling_rate

                if evals >= self.budget:
                    break

            population = np.array(new_population)
            if len(population) < population_size:
                additional_individuals = np.random.uniform(lower_bound, upper_bound, (population_size - len(population), self.dim))
                population = np.vstack((population, additional_individuals))

        return best_solution