import numpy as np

class ImprovedEnhancedEAPTEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_solution = None
        best_value = float('inf')

        initial_pop_size = 10 + 2 * self.dim
        max_pop_size = initial_pop_size * 2
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (initial_pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.eval_count += initial_pop_size
        
        memory = []

        def local_search(x, adapt_strength):
            mutation_var = adapt_strength * np.random.randn(self.dim)
            candidate = x + mutation_var
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate
        
        def hybrid_mutation(x):
            noise = np.random.randn(self.dim) * (0.1 + 0.9 * np.random.rand()) * (1 - (self.eval_count / self.budget))
            candidate = x + noise
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate

        def adaptive_crossover(a, b, c):
            F = 0.3 + 0.7 * np.random.rand() * (1 - (self.eval_count / self.budget))
            candidate = a + F * (b - c)
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate

        while self.eval_count < self.budget:
            adapt_strength = 0.5 + 0.5 * (1 - (self.eval_count / self.budget))
            
            for i in range(len(population)):
                if self.eval_count >= self.budget:
                    break
                
                if np.random.rand() < 0.35:  # Reduce random exploration rate
                    new_solution = np.random.uniform(bounds[:, 0], bounds[:, 1], self.dim)
                else:
                    if memory and np.random.rand() < 0.25:  # Adjusted memory usage rate
                        mem_index = np.random.choice(len(memory))
                        new_solution = hybrid_mutation(memory[mem_index])
                    elif len(population) >= 4:  # Use four points for crossover
                        idxs = np.random.choice(len(population), 4, replace=False)
                        a, b, c, d = population[idxs]
                        new_solution = adaptive_crossover((a+b)/2, (c+d)/2, a)
                    else:
                        new_solution = local_search(population[i], adapt_strength)
                
                new_value = func(new_solution)
                self.eval_count += 1
                
                if new_value < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_value
                    memory.append(new_solution)

                    if len(memory) > max_pop_size * 0.75:  # Dynamically manage memory size
                        memory.pop(0)
                
                if new_value < best_value:
                    best_solution = new_solution
                    best_value = new_value

            if len(population) < max_pop_size and np.random.rand() < 0.15:  # Adjust population expansion rate
                additional = np.random.uniform(bounds[:, 0], bounds[:, 1], (initial_pop_size // 2, self.dim))
                pop_fitness = np.array([func(x) for x in additional])
                self.eval_count += len(additional)
                population = np.vstack((population, additional))
                fitness = np.hstack((fitness, pop_fitness))

        return best_solution, best_value