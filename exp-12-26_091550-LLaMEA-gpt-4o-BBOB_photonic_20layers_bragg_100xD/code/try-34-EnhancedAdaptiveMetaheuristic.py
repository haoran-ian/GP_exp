import numpy as np

class EnhancedAdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, dim * 2)
        self.learning_rate = 0.1
        self.strategy_pool = [self.random_search, self.gradient_search, self.crossover_mutation, self.dynamic_local_search]
        self.strategy_probabilities = np.ones(len(self.strategy_pool)) / len(self.strategy_pool)
        self.success_rates = np.zeros(len(self.strategy_pool))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            strategy_idx = np.random.choice(len(self.strategy_pool), p=self.strategy_probabilities)
            new_population = self.strategy_pool[strategy_idx](population, func, lb, ub)
            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += len(new_population)
            
            # Update best solution
            new_best_idx = np.argmin(new_fitness)
            if new_fitness[new_best_idx] < best_fitness:
                best_fitness = new_fitness[new_best_idx]
                best_solution = new_population[new_best_idx].copy()
            
            # Adaptive strategy selection using reinforced learning
            success = new_fitness < fitness
            self.success_rates[strategy_idx] += success.mean()
            self.strategy_probabilities = self.success_rates + 1e-2  # Avoid zero division
            self.strategy_probabilities /= self.strategy_probabilities.sum()
            
            # Move to the next generation
            population, fitness = new_population, new_fitness
        
        return best_solution

    def random_search(self, population, func, lb, ub):
        return np.random.uniform(lb, ub, population.shape)
    
    def gradient_search(self, population, func, lb, ub):
        perturbation = np.random.randn(*population.shape) * 0.1
        candidates = population + perturbation
        np.clip(candidates, lb, ub, out=candidates)
        return candidates
    
    def crossover_mutation(self, population, func, lb, ub):
        offspring = []
        for _ in range(self.population_size):
            parents = population[np.random.choice(len(population), 2, replace=False)]
            cross_point = np.random.randint(1, self.dim - 1)
            child = np.concatenate((parents[0][:cross_point], parents[1][cross_point:]))
            mutation = np.random.randn(self.dim) * 0.05
            child += mutation
            np.clip(child, lb, ub, out=child)
            offspring.append(child)
        return np.array(offspring)

    def dynamic_local_search(self, population, func, lb, ub):
        # Dynamic neighborhood exploration
        neighborhood_size = max(1, int(0.1 * self.dim))
        improved_population = []
        for ind in population:
            local_best = ind
            local_best_fitness = func(ind)
            for _ in range(neighborhood_size):
                candidate = ind + np.random.uniform(-0.1, 0.1, self.dim)
                np.clip(candidate, lb, ub, out=candidate)
                candidate_fitness = func(candidate)
                if candidate_fitness < local_best_fitness:
                    local_best = candidate
                    local_best_fitness = candidate_fitness
            improved_population.append(local_best)
        return np.array(improved_population)