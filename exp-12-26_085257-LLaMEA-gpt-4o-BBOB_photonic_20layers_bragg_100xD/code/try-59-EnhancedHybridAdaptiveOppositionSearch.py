import numpy as np

class EnhancedHybridAdaptiveOppositionSearch:
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
        C = np.eye(self.dim)
        sigma = 0.3
        
        def levy_flight(Lambda):
            u = np.random.normal(0, 1, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / np.power(np.abs(v), 1/Lambda)
            return step

        while evals < self.budget:
            reflect_pop = bounds[0] + bounds[1] - population
            new_pop = population + levy_flight(1.5).dot(C)
            eval_pop = np.vstack((reflect_pop, new_pop))
            eval_fitness = np.apply_along_axis(func, 1, eval_pop)
            evals += eval_pop.shape[0]

            combined_population = np.vstack((population, reflect_pop, new_pop))
            combined_fitness = np.hstack((fitness, eval_fitness))
            sorted_indices = np.argsort(combined_fitness)
            population = combined_population[sorted_indices][:pop_size]
            fitness = combined_fitness[sorted_indices][:pop_size]

            if fitness[0] < best_fitness:
                best_solution = population[0]
                best_fitness = fitness[0]

            perturbation = np.random.multivariate_normal(np.zeros(self.dim), C * sigma, pop_size)
            population += perturbation
            population = np.clip(population, bounds[0], bounds[1])

            fitness = np.apply_along_axis(func, 1, population)
            evals += pop_size

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]

            # Update covariance matrix using successful steps
            successful_steps = perturbation[fitness < best_fitness]
            if len(successful_steps) > 0:
                mean_step = np.mean(successful_steps, axis=0)
                C = (1 - 0.2) * C + 0.2 * np.outer(mean_step, mean_step)
                sigma *= 0.9  # Decrease step size for finer search

        return best_solution