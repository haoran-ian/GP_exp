import numpy as np

class DualPhaseAdaptiveSearch:
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

        exploration_phase_end = int(0.6 * self.budget)
        
        while evals < self.budget:
            if evals < exploration_phase_end:
                # Reflective opposition for exploration
                reflect_pop = bounds[0] + bounds[1] - population
                reflect_fitness = np.apply_along_axis(func, 1, reflect_pop)
                evals += pop_size
                
                # Combine and select
                combined_population = np.vstack((population, reflect_pop))
                combined_fitness = np.hstack((fitness, reflect_fitness))
                sorted_indices = np.argsort(combined_fitness)
                population = combined_population[sorted_indices][:pop_size]
                fitness = combined_fitness[sorted_indices][:pop_size]
                
                if fitness[0] < best_fitness:
                    best_solution = population[0]
                    best_fitness = fitness[0]
                
                # Lévy flight for enhanced exploration
                for i in range(pop_size):
                    levy_step = levy_flight(1.5)
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
            else:
                # Adaptive Neighborhood Search for exploitation
                step_size = 0.05 * (bounds[1] - bounds[0]) * ((self.budget - evals) / (self.budget - exploration_phase_end))
                perturbation = np.random.normal(0, step_size, population.shape)
                population += perturbation
                population = np.clip(population, bounds[0], bounds[1])
                
                # Evaluate the perturbed population
                fitness = np.apply_along_axis(func, 1, population)
                evals += pop_size
                
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < best_fitness:
                    best_solution = population[best_idx]
                    best_fitness = fitness[best_idx]

        return best_solution