import numpy as np

class QuantumEAPTEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_solution = None
        best_value = float('inf')

        initial_pop_size = 15 + 3 * self.dim
        max_pop_size = initial_pop_size * 3
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (initial_pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.eval_count += initial_pop_size
        
        memory = []

        def quantum_mutation(x, adapt_strength):
            alpha = 0.1 + 0.9 * np.random.rand()
            direction = np.random.randn(self.dim)
            step = adapt_strength * np.tanh(np.random.randn()) * direction
            candidate = x + step
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate

        def stochastic_neighborhood_search(x):
            direction = np.random.normal(size=self.dim)
            strength = np.random.uniform(0.1, 1.0)
            candidate = x + strength * direction
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate

        def adaptive_entropy_crossover(a, b, c):
            entropy_coef = np.std(population, axis=0).mean()
            F = 0.2 + entropy_coef * (1 - (self.eval_count / self.budget))
            candidate = a + F * (b - c)
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate

        while self.eval_count < self.budget:
            adapt_strength = 0.4 + 0.6 * (1 - (self.eval_count / self.budget))
            
            for i in range(len(population)):
                if self.eval_count >= self.budget:
                    break

                if np.random.rand() < 0.5:
                    new_solution = stochastic_neighborhood_search(population[i])
                else:
                    if memory and np.random.rand() < 0.4:
                        mem_index = np.random.choice(len(memory))
                        new_solution = quantum_mutation(memory[mem_index], adapt_strength)
                    elif len(population) >= 3:
                        idxs = np.random.choice(len(population), 3, replace=False)
                        a, b, c = population[idxs]
                        new_solution = adaptive_entropy_crossover(a, b, c)
                    else:
                        new_solution = quantum_mutation(population[i], adapt_strength)
                
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

            if len(population) < max_pop_size and np.random.rand() < 0.3:
                additional = np.random.uniform(bounds[:, 0], bounds[:, 1], (initial_pop_size // 2, self.dim))
                pop_fitness = np.array([func(x) for x in additional])
                self.eval_count += len(additional)
                population = np.vstack((population, additional))
                fitness = np.hstack((fitness, pop_fitness))

        return best_solution, best_value