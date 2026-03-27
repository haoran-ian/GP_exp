import numpy as np

class EnhancedEAPTEAlgorithmV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_solution = None
        best_value = float('inf')

        initial_pop_size = 10 + 2 * self.dim
        max_pop_size = initial_pop_size * 3
        min_pop_size = initial_pop_size // 2
        population = self._initialize_population(bounds, initial_pop_size)
        fitness = np.array([func(x) for x in population])
        self.eval_count += initial_pop_size
        
        memory = []

        def local_search(x, adapt_strength):
            mutation_var = adapt_strength * np.random.randn(self.dim)
            candidate = x + mutation_var
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate
        
        def chaotic_mutation(x):
            beta = np.random.rand()
            noise = np.random.randn(self.dim) * (0.1 + 0.9 * beta) * (1 - (self.eval_count / self.budget))
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
                
                if np.random.rand() < 0.4:
                    new_solution = np.random.uniform(bounds[:, 0], bounds[:, 1], self.dim)
                else:
                    if memory and np.random.rand() < 0.3:
                        mem_index = np.random.choice(len(memory))
                        new_solution = chaotic_mutation(memory[mem_index])
                    elif len(population) >= 3:
                        idxs = np.random.choice(len(population), 3, replace=False)
                        a, b, c = population[idxs]
                        new_solution = adaptive_crossover(a, b, c)
                    else:
                        new_solution = local_search(population[i], adapt_strength)
                
                new_value = func(new_solution)
                self.eval_count += 1
                
                if new_value < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_value
                    memory.append(new_solution)
                    
                    if len(memory) > max_pop_size:
                        memory.pop(np.random.choice(len(memory)))  # Adaptive memory decay

                if new_value < best_value:
                    best_solution = new_solution
                    best_value = new_value

            current_budget_ratio = self.eval_count / self.budget
            dynamic_pop_size = int(min_pop_size + (max_pop_size - min_pop_size) * (1 - current_budget_ratio))
            if len(population) < dynamic_pop_size and np.random.rand() < 0.2:
                additional = np.random.uniform(bounds[:, 0], bounds[:, 1], (dynamic_pop_size - len(population), self.dim))
                pop_fitness = np.array([func(x) for x in additional])
                self.eval_count += len(additional)
                population = np.vstack((population, additional))
                fitness = np.hstack((fitness, pop_fitness))

        return best_solution, best_value

    def _initialize_population(self, bounds, size):
        # Chaotic initialization using logistic map
        logistic_map = lambda x: 4 * x * (1 - x)
        chaos = np.random.rand(size, self.dim)
        for i in range(100):  # Iterate to enhance chaos properties
            chaos = logistic_map(chaos)
        population = bounds[:, 0] + chaos * (bounds[:, 1] - bounds[:, 0])
        return population