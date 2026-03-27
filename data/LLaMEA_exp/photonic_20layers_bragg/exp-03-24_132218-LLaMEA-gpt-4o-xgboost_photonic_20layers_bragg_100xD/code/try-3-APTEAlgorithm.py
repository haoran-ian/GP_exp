import numpy as np

class APTEAlgorithm:
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
        
        # Adaptive memory to store recently improved solutions
        memory = []
        
        # Elite selection
        elite_size = max(1, pop_size // 5)
        
        def local_search(x, adapt_strength):
            # Conduct local search to refine solutions around a point
            candidate = x + np.random.randn(self.dim) * adapt_strength
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate

        while self.eval_count < self.budget:
            adapt_strength = 0.3 + 0.7 * (1 - (self.eval_count / self.budget))
            
            # Dynamic mutation strategy based on fitness variability
            fitness_variance = np.var(fitness)
            dynamic_mutation_strength = adapt_strength * (1 + fitness_variance)
            
            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break
                
                if np.random.rand() < 0.5:
                    # Exploration Phase
                    new_solution = np.random.uniform(bounds[:, 0], bounds[:, 1], self.dim)
                else:
                    # Exploitation Phase or Memory-Driven Search
                    if memory and np.random.rand() < 0.3:
                        mem_index = np.random.choice(len(memory))
                        new_solution = local_search(memory[mem_index], dynamic_mutation_strength)
                    else:
                        new_solution = local_search(population[i], dynamic_mutation_strength)
                
                new_value = func(new_solution)
                self.eval_count += 1
                
                # Update population if new solution is better
                if new_value < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_value
                    memory.append(new_solution)  # Store in memory
                    
                    # Maintain memory size limit
                    if len(memory) > pop_size:
                        memory.pop(0)
                
                # Update the best found solution
                if new_value < best_value:
                    best_solution = new_solution
                    best_value = new_value

            # Elite selection: keep the top-performing solutions
            elite_indices = fitness.argsort()[:elite_size]
            population = population[elite_indices]
            fitness = fitness[elite_indices]
            
            # Replenish the population with new random solutions
            new_population = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size - elite_size, self.dim))
            population = np.vstack((population, new_population))
            new_fitness = np.array([func(x) for x in new_population])
            fitness = np.concatenate((fitness, new_fitness))
            self.eval_count += len(new_population)

        return best_solution, best_value