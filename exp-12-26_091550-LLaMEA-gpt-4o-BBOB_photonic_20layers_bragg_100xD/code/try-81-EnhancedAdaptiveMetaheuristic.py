import numpy as np

class EnhancedAdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, dim * 3)
        self.learning_rate = 0.1
        self.strategy_pool = [self.random_search, self.gradient_search, self.crossover_mutation]
        self.strategy_probabilities = np.ones(len(self.strategy_pool)) / len(self.strategy_pool)
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            strategy_idx = np.random.choice(len(self.strategy_pool), p=self.strategy_probabilities)
            new_population = self.strategy_pool[strategy_idx](population, func, lb, ub)
            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += len(new_population)
            
            # Update and preserve elite solution
            self.update_best_solution(new_population, new_fitness)
            
            # Adaptive strategy selection with dynamic learning rate
            success_rate = (new_fitness < fitness).mean()
            self.strategy_probabilities[strategy_idx] += self.learning_rate * success_rate
            self.strategy_probabilities /= self.strategy_probabilities.sum()
            
            # Move to the next generation with elite preservation
            elite_indices = np.argsort(fitness)[:max(1, self.population_size // 10)]
            elite_population = population[elite_indices]
            elite_fitness = fitness[elite_indices]
            population = np.vstack((new_population, elite_population))
            fitness = np.hstack((new_fitness, elite_fitness))
            best_idx = np.argmin(fitness)
            population = population[np.argsort(fitness)][:self.population_size]
            fitness = fitness[np.argsort(fitness)][:self.population_size]
        
        return self.best_solution

    def update_best_solution(self, population, fitness):
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_solution = population[best_idx].copy()

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