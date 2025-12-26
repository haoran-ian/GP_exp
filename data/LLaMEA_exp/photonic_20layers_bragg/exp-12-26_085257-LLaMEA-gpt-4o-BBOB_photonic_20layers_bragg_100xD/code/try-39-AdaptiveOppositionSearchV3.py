import numpy as np

class AdaptiveOppositionSearchV3:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        pop_size = min(50, self.budget // 10)
        num_subpopulations = 3
        subpop_sizes = [pop_size // num_subpopulations] * num_subpopulations
        populations = [np.random.uniform(bounds[0], bounds[1], (size, self.dim)) for size in subpop_sizes]
        fitness = [np.apply_along_axis(func, 1, pop) for pop in populations]
        best_idx = np.argmin([f.min() for f in fitness])
        best_solution = populations[best_idx][np.argmin(fitness[best_idx])]
        best_fitness = fitness[best_idx].min()
        evals = sum(subpop_sizes)

        def levy_flight(Lambda):
            u = np.random.normal(0, 1, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / np.power(np.abs(v), 1/Lambda)
            return step

        while evals < self.budget:
            for idx in range(num_subpopulations):
                # Reflective opposition
                reflect_pop = bounds[0] + bounds[1] - populations[idx]
                reflect_fitness = np.apply_along_axis(func, 1, reflect_pop)
                evals += subpop_sizes[idx]
                
                # Combine and select
                combined_population = np.vstack((populations[idx], reflect_pop))
                combined_fitness = np.hstack((fitness[idx], reflect_fitness))
                sorted_indices = np.argsort(combined_fitness)
                populations[idx] = combined_population[sorted_indices][:subpop_sizes[idx]]
                fitness[idx] = combined_fitness[sorted_indices][:subpop_sizes[idx]]
                
                if fitness[idx][0] < best_fitness:
                    best_solution = populations[idx][0]
                    best_fitness = fitness[idx][0]
                
                # Dynamic mutation rate
                mutation_rate = (1 - (evals / self.budget)) * 0.1
                step_size = mutation_rate * (bounds[1] - bounds[0])
                perturbation = np.random.normal(0, step_size, populations[idx].shape)
                populations[idx] += perturbation
                populations[idx] = np.clip(populations[idx], bounds[0], bounds[1])
                
                # Lévy flight for enhanced exploration
                for i in range(subpop_sizes[idx]):
                    levy_step = levy_flight(1.5)
                    child = populations[idx][i] + levy_step
                    child = np.clip(child, bounds[0], bounds[1])
                    child_fitness = func(child)
                    evals += 1
                    if child_fitness < fitness[idx][i]:
                        populations[idx][i] = child
                        fitness[idx][i] = child_fitness
                        if child_fitness < best_fitness:
                            best_solution = child
                            best_fitness = child_fitness
                
                if evals >= self.budget:
                    break

            if evals >= self.budget:
                break
            
            # Evaluate all populations
            for idx in range(num_subpopulations):
                fitness[idx] = np.apply_along_axis(func, 1, populations[idx])
                evals += subpop_sizes[idx]
                
                best_idx = np.argmin(fitness[idx])
                if fitness[idx][best_idx] < best_fitness:
                    best_solution = populations[idx][best_idx]
                    best_fitness = fitness[idx][best_idx]

        return best_solution