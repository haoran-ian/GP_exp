import numpy as np

class EnhancedAdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, dim * 2)
        self.learning_rate = 0.1
        self.strategy_pool = [self.random_search, self.gradient_search, self.crossover_mutation, self.gradient_descent]
        self.strategy_probabilities = np.ones(len(self.strategy_pool)) / len(self.strategy_pool)
        self.local_search_intensity = 0.05
    
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
            
            # Adaptive strategy selection
            success_rate = np.sum(new_fitness < fitness) / len(fitness)
            self.strategy_probabilities[strategy_idx] += self.learning_rate * success_rate
            self.strategy_probabilities /= self.strategy_probabilities.sum()
            
            # Move to the next generation
            population, fitness = new_population, new_fitness
        
        return best_solution

    def random_search(self, population, func, lb, ub):
        return np.random.uniform(lb, ub, population.shape)
    
    def gradient_search(self, population, func, lb, ub):
        perturbation = np.random.randn(*population.shape) * self.local_search_intensity
        candidates = population + perturbation
        np.clip(candidates, lb, ub, out=candidates)
        return candidates
    
    def crossover_mutation(self, population, func, lb, ub):
        offspring = []
        for _ in range(self.population_size):
            parents = population[np.random.choice(len(population), 2, replace=False)]
            cross_point = np.random.randint(1, self.dim - 1)
            child = np.concatenate((parents[0][:cross_point], parents[1][cross_point:]))
            mutation = np.random.randn(self.dim) * self.local_search_intensity
            child += mutation
            np.clip(child, lb, ub, out=child)
            offspring.append(child)
        return np.array(offspring)

    def gradient_descent(self, population, func, lb, ub):
        gradient_step = 0.01
        gradients = np.zeros_like(population)
        for i, ind in enumerate(population):
            original_value = func(ind)
            for j in range(self.dim):
                ind[j] += gradient_step
                if ind[j] > ub[j]:
                    ind[j] = ub[j]
                elif ind[j] < lb[j]:
                    ind[j] = lb[j]
                new_value = func(ind)
                gradients[i, j] = (new_value - original_value) / gradient_step
                ind[j] -= gradient_step  # restore
            
            gradients[i] /= np.linalg.norm(gradients[i]) + 1e-8
            population[i] -= self.local_search_intensity * gradients[i]
            np.clip(population[i], lb, ub, out=population[i])
        
        return population