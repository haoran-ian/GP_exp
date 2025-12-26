import numpy as np

class MultiScaleAdaptiveOpposition:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        pop_size = min(50, self.budget // 10)
        population = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        evals = pop_size
        memory = np.copy(population[:2])  # Initialize with top 2 solutions

        def levy_flight(Lambda):
            u = np.random.normal(0, 1, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / np.power(np.abs(v), 1 / Lambda)
            return step

        while evals < self.budget:
            reflect_pop = bounds[0] + bounds[1] - population
            new_pop = population + levy_flight(1.5)
            eval_pop = np.vstack((reflect_pop, new_pop))
            eval_fitness = np.apply_along_axis(func, 1, eval_pop)
            evals += eval_pop.shape[0]

            combined_population = np.vstack((population, reflect_pop, new_pop))
            combined_fitness = np.hstack((fitness, eval_fitness))
            sorted_indices = np.argsort(combined_fitness)
            population = combined_population[sorted_indices][:pop_size]
            fitness = combined_fitness[sorted_indices][:pop_size]

            # Dynamic recombination and contextual memory usage
            if evals < self.budget // 2:
                diversity_factor = 0.5
            else:
                diversity_factor = 1.0
            memory_recombination = (memory + population[:2]) / 2
            memory_recombination += np.random.normal(0, diversity_factor, memory_recombination.shape)

            population[:2] = memory_recombination
            memory = np.copy(population[:2])

            step_size = 0.03 * (bounds[1] - bounds[0])
            perturbation = np.random.normal(0, step_size, population.shape)
            population += perturbation
            population = np.clip(population, bounds[0], bounds[1])

            fitness = np.apply_along_axis(func, 1, population)
            evals += pop_size

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]

        return best_solution