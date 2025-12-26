import numpy as np

class EnhancedOppositionSearch:
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

        def levy_flight(Lambda):
            u = np.random.normal(0, 1, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / np.power(np.abs(v), 1/Lambda)
            return 0.01 * step  # Reduced scale for finer exploration

        def chaotic_map(x):
            return 4 * x * (1 - x)

        chaos_param = np.random.rand()

        while evals < self.budget:
            reflect_pop = bounds[0] + bounds[1] - population
            reflect_fitness = np.apply_along_axis(func, 1, reflect_pop)
            evals += pop_size

            combined_population = np.vstack((population, reflect_pop))
            combined_fitness = np.hstack((fitness, reflect_fitness))
            sorted_indices = np.argsort(combined_fitness)
            population = combined_population[sorted_indices][:pop_size]
            fitness = combined_fitness[sorted_indices][:pop_size]

            if fitness[0] < best_fitness:
                best_solution = population[0]
                best_fitness = fitness[0]

            step_size = 0.02 * (bounds[1] - bounds[0])  # Further decreased step size
            perturbation = np.random.normal(0, step_size, population.shape)
            population += perturbation
            population = np.clip(population, bounds[0], bounds[1])

            for i in range(pop_size):
                chaos_param = chaotic_map(chaos_param)
                levy_step = chaos_param * levy_flight(1.7)  # Chaotic Lévy flight
                child = population[i] + levy_step
                child = np.clip(child, bounds[0], bounds[1])
                child_fitness = func(child)
                evals += 1
                if child_fitness < fitness[i]:
                    population[i] = child
                    fitness[i] = child_fitness
                    if child_fitness < best_fitness:
                        best_solution = child
                        best_fitness = child_fitness

            if evals >= self.budget:
                break

            fitness = np.apply_along_axis(func, 1, population)
            evals += pop_size

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]
                
        return best_solution