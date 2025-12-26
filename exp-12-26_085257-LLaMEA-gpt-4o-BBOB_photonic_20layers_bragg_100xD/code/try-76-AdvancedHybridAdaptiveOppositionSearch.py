import numpy as np

class AdvancedHybridAdaptiveOppositionSearch:
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

        def dynamic_exploration_exploitation_rate(evals):
            return 0.5 + 0.5 * np.cos(np.pi * evals / self.budget)

        while evals < self.budget:
            exploration_rate = dynamic_exploration_exploitation_rate(evals)
            reflect_pop = bounds[0] + bounds[1] - population
            new_pop = population + levy_flight(1.5) * exploration_rate
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
            
            step_size = 0.03 * (bounds[1] - bounds[0]) * exploration_rate
            perturbation = np.random.normal(0, step_size, population.shape)
            population += perturbation
            population = np.clip(population, bounds[0], bounds[1])
            
            population[:2] = combined_population[sorted_indices][:2]
            
            fitness = np.apply_along_axis(func, 1, population)
            evals += pop_size
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]
                
            breakthrough_learning_rate = 0.02
            for i in range(1, pop_size):
                if np.random.rand() < breakthrough_learning_rate:
                    population[i] = population[0] + np.random.normal(0, step_size, self.dim)
                    population[i] = np.clip(population[i], bounds[0], bounds[1])
                    fitness[i] = func(population[i])
                    evals += 1
                    if fitness[i] < best_fitness:
                        best_solution = population[i]
                        best_fitness = fitness[i]

        return best_solution