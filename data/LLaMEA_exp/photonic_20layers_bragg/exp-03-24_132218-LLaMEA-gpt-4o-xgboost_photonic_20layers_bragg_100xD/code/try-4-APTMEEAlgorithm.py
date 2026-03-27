import numpy as np

class APTMEEAlgorithm:
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

        def local_search(x, adapt_strength):
            # Conduct local search to refine solutions around a point
            candidate = x + np.random.randn(self.dim) * adapt_strength
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate

        while self.eval_count < self.budget:
            adapt_strength = 0.2 + 0.8 * (1 - (self.eval_count / self.budget))  # More pronounced adaptive behavior
            
            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break
                
                if np.random.rand() < 0.5:
                    # Exploration Phase
                    new_solution = np.random.uniform(bounds[:, 0], bounds[:, 1], self.dim)
                else:
                    # Exploitation Phase or Memory-Driven Search
                    if memory and np.random.rand() < 0.4:  # Higher chance to use memory
                        # Select memory based on fitness rank
                        mem_index = np.random.choice(len(memory), p=np.linspace(1, 0, len(memory))/np.sum(np.linspace(1, 0, len(memory))))
                        new_solution = local_search(memory[mem_index], adapt_strength)
                    else:
                        new_solution = local_search(population[i], adapt_strength)
                
                new_value = func(new_solution)
                self.eval_count += 1
                
                # Update population if new solution is better
                if new_value < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_value
                    memory.append(new_solution)  # Store in memory

                    # Maintain memory size limit and prioritize recent better solutions
                    if len(memory) > pop_size:
                        memory.pop(0)
                
                # Update the best found solution
                if new_value < best_value:
                    best_solution = new_solution
                    best_value = new_value

        return best_solution, best_value