import numpy as np

class EAPTEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_solution = None
        best_value = float('inf')
        
        # Initialize population
        pop_size = 10 + 2 * self.dim
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.eval_count += pop_size
        
        # Memory of best historical position
        historical_best = np.copy(population)
        historical_fitness = np.copy(fitness)
        
        def local_search(x, scale):
            # Conduct local search to refine solutions around a point
            perturbation_strength = scale
            candidate = x + np.random.randn(self.dim) * perturbation_strength
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate

        while self.eval_count < self.budget:
            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break
                
                # Dynamic selection of strategy based on local landscape
                phase_switch_prob = 0.3 + 0.7 * (fitness[i] - best_value) / (np.max(fitness) - np.min(fitness) + 1e-9)
                if np.random.rand() < phase_switch_prob:
                    # Exploration Phase
                    new_solution = np.random.uniform(bounds[:, 0], bounds[:, 1], self.dim)
                else:
                    # Exploitation Phase with adaptive perturbation
                    scale = 0.1 + 0.4 * (fitness[i] - best_value) / (np.max(fitness) - np.min(fitness) + 1e-9)
                    new_solution = local_search(population[i], scale)
                
                new_value = func(new_solution)
                self.eval_count += 1
                
                # Update population if new solution is better
                if new_value < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_value
                
                # Update historical best if new solution is better
                if new_value < historical_fitness[i]:
                    historical_best[i] = new_solution
                    historical_fitness[i] = new_value
                
                # Update the best found solution
                if new_value < best_value:
                    best_solution = new_solution
                    best_value = new_value
            
            # Occasionally replace worst solutions with historical bests
            if self.eval_count < self.budget - pop_size:
                worst_indices = np.argsort(fitness)[-int(pop_size * 0.2):]
                for idx in worst_indices:
                    if self.eval_count >= self.budget:
                        break
                    population[idx] = historical_best[idx]
                    fitness[idx] = historical_fitness[idx]
                    self.eval_count += 1

        return best_solution, best_value