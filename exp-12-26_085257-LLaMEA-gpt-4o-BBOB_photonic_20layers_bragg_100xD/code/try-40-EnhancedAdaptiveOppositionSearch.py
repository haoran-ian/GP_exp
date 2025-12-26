import numpy as np

class EnhancedAdaptiveOppositionSearch:
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

        Lambda = 1.5
        
        while evals < self.budget:
            # Reflective opposition
            reflect_pop = bounds[0] + bounds[1] - population
            reflect_fitness = np.apply_along_axis(func, 1, reflect_pop)
            evals += pop_size
            
            # Combine and select using stochastic ranking
            combined_population = np.vstack((population, reflect_pop))
            combined_fitness = np.hstack((fitness, reflect_fitness))
            ranks = np.argsort(np.argsort(combined_fitness))
            rank_prob = np.random.uniform(0, 1, rank_prob.shape)
            selection_criteria = rank_prob < 0.45  # Stochastic selection probability
            selected_indices = np.argsort(combined_fitness + rank_prob)[:pop_size]
            population = combined_population[selected_indices]
            fitness = combined_fitness[selected_indices]
            
            if fitness[0] < best_fitness:
                best_solution = population[0]
                best_fitness = fitness[0]
            
            # Dynamic local search intensification
            step_size = 0.1 * (bounds[1] - bounds[0])
            perturbation = np.random.normal(0, step_size, population.shape)
            population += perturbation
            population = np.clip(population, bounds[0], bounds[1])
            
            # Adaptive Lévy flight for enhanced exploration
            for i in range(pop_size):
                if np.random.rand() < 0.3:  # Adjust Lambda to balance exploration
                    Lambda = 1.3 + np.random.rand()
                levy_step = levy_flight(Lambda)
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