import numpy as np

class QuantumTunnelingHybridSearch:
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
            return step

        def quantum_tunneling(x, best):
            tunneling_prob = np.random.rand(self.dim)
            tunnel = np.where(tunneling_prob < 0.5, best - (best - x) * np.random.rand(self.dim), x)
            return tunnel

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
            
            if fitness[0] < best_fitness:
                best_solution = population[0]
                best_fitness = fitness[0]

            step_size = 0.03 * (bounds[1] - bounds[0])
            perturbation = np.random.normal(0, step_size, population.shape)
            population += perturbation
            population = np.clip(population, bounds[0], bounds[1])

            population[:2] = combined_population[sorted_indices][:2]
            
            tunneling_population = np.array([quantum_tunneling(x, best_solution) for x in population])
            tunneling_fitness = np.apply_along_axis(func, 1, tunneling_population)
            evals += pop_size
            
            combined_population = np.vstack((population, tunneling_population))
            combined_fitness = np.hstack((fitness, tunneling_fitness))
            sorted_indices = np.argsort(combined_fitness)
            population = combined_population[sorted_indices][:pop_size]
            fitness = combined_fitness[sorted_indices][:pop_size]

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]

        return best_solution