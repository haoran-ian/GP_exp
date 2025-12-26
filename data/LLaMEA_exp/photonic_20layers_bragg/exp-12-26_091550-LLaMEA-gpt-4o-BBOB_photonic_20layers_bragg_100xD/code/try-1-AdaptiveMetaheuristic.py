import numpy as np

class AdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, dim * 2)
        self.learning_rate = 0.1
        self.strategy_pool = [self.random_search, self.gradient_search, self.crossover_mutation]
        self.strategy_probabilities = np.ones(len(self.strategy_pool)) / len(self.strategy_pool)
        self.memory = []  # Adaptive memory for storing good solutions

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
                # Add to memory if it's significantly better
                if len(self.memory) == 0 or (best_fitness < min(x[1] for x in self.memory) * 0.95):
                    self.memory.append((best_solution.copy(), best_fitness))
            
            # Adaptive strategy selection with memory analysis
            success = new_fitness < fitness
            self.strategy_probabilities[strategy_idx] = max(0.1, self.strategy_probabilities[strategy_idx] + self.learning_rate * success.mean())
            self.strategy_probabilities /= self.strategy_probabilities.sum()
            
            # Use memory to refine exploration-exploitation balance
            if len(self.memory) > 0 and evaluations < self.budget * 0.8:
                memory_idx = np.random.choice(len(self.memory))
                memory_solution = self.memory[memory_idx][0]
                new_population = np.vstack([new_population, memory_solution + np.random.randn(self.dim) * 0.01])
                new_fitness = np.append(new_fitness, func(new_population[-1]))
                evaluations += 1
            
            # Move to the next generation
            combined_population = np.vstack([population, new_population])
            combined_fitness = np.append(fitness, new_fitness)
            selected_indices = np.argsort(combined_fitness)[:self.population_size]
            population, fitness = combined_population[selected_indices], combined_fitness[selected_indices]
        
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