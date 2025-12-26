import numpy as np

class EnhancedAdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, dim * 2)
        self.learning_rate = 0.1
        self.strategy_pool = [self.random_search, self.gradient_search, self.crossover_mutation]
        self.strategy_probabilities = np.ones(len(self.strategy_pool)) / len(self.strategy_pool)
        self.memory = np.zeros(len(self.strategy_pool))
        self.parameter_pool = [0.1, 0.05, 0.01]
        self.param_selection_probability = np.ones(len(self.parameter_pool)) / len(self.parameter_pool)
        
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
            param_idx = np.random.choice(len(self.parameter_pool), p=self.param_selection_probability)
            new_population = self.strategy_pool[strategy_idx](population, func, lb, ub, self.parameter_pool[param_idx])
            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += len(new_population)
            
            # Update best solution
            new_best_idx = np.argmin(new_fitness)
            if new_fitness[new_best_idx] < best_fitness:
                best_fitness = new_fitness[new_best_idx]
                best_solution = new_population[new_best_idx].copy()
                self.memory[strategy_idx] += 1  # Reward successful strategy
            
            # Adaptive strategy selection with memory
            success = new_fitness < fitness
            self.strategy_probabilities[strategy_idx] += self.learning_rate * success.mean()
            self.strategy_probabilities += self.memory  # Bias towards historically successful strategies
            self.strategy_probabilities /= self.strategy_probabilities.sum()
            
            # Self-adaptive parameter tuning
            self.param_selection_probability[param_idx] += self.learning_rate * success.mean()
            self.param_selection_probability /= self.param_selection_probability.sum()
            
            # Move to the next generation
            population, fitness = new_population, new_fitness
        
        return best_solution

    def random_search(self, population, func, lb, ub, param):
        return np.random.uniform(lb, ub, population.shape)
    
    def gradient_search(self, population, func, lb, ub, param):
        perturbation = np.random.randn(*population.shape) * param
        candidates = population + perturbation
        np.clip(candidates, lb, ub, out=candidates)
        return candidates
    
    def crossover_mutation(self, population, func, lb, ub, param):
        offspring = []
        for _ in range(self.population_size):
            parents = population[np.random.choice(len(population), 2, replace=False)]
            cross_point = np.random.randint(1, self.dim - 1)
            child = np.concatenate((parents[0][:cross_point], parents[1][cross_point:]))
            mutation = np.random.randn(self.dim) * param
            child += mutation
            np.clip(child, lb, ub, out=child)
            offspring.append(child)
        return np.array(offspring)