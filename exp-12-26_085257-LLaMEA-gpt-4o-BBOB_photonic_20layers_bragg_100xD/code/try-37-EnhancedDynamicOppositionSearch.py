import numpy as np

class EnhancedDynamicOppositionSearch:
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
            scale = np.random.uniform(0.1, 1.0)  # Adaptive scaling factor
            return scale * step

        while evals < self.budget:
            # Reflective opposition
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

            # Differential evolution-inspired mutation
            F = 0.8  # Differential weight
            for i in range(pop_size):
                candidates = range(pop_size)
                a, b, c = population[np.random.choice(candidates, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])
                mutant_fitness = func(mutant)
                evals += 1
                if mutant_fitness < fitness[i]:
                    population[i] = mutant
                    fitness[i] = mutant_fitness
                    if mutant_fitness < best_fitness:
                        best_solution = mutant
                        best_fitness = mutant_fitness

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
            
            if evals >= self.budget:
                break

            # Evaluate the perturbed population 
            fitness = np.apply_along_axis(func, 1, population)
            evals += pop_size
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]
                
        return best_solution