import numpy as np

class EnhancedEAPTEAlgorithm:
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

        def stochastic_local_exploitation(x, scale):
            noise = scale * np.random.randn(self.dim)
            candidate = x + noise
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate
        
        def dynamic_crossover(a, b, c):
            F = 0.5 + 0.5 * np.random.rand() * (1 - (self.eval_count / self.budget))
            candidate = a + F * (b - c)
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate

        def update_population():
            if len(population) < max_pop_size and np.random.rand() < 0.3:
                increase_size = min(max_pop_size - len(population), initial_pop_size // 2)
                additional = np.random.uniform(bounds[:, 0], bounds[:, 1], (increase_size, self.dim))
                pop_fitness = np.array([func(x) for x in additional])
                self.eval_count += len(additional)
                return np.vstack((population, additional)), np.hstack((fitness, pop_fitness))
            return population, fitness

        while self.eval_count < self.budget:
            scale_factor = 0.3 + 0.7 * (1 - (self.eval_count / self.budget))
            
            for i in range(len(population)):
                if self.eval_count >= self.budget:
                    break
                
                if np.random.rand() < 0.3:
                    new_solution = stochastic_local_exploitation(population[i], scale_factor)
                else:
                    if memory and np.random.rand() < 0.25:
                        mem_index = np.random.choice(len(memory))
                        new_solution = stochastic_local_exploitation(memory[mem_index], scale_factor)
                    elif len(population) >= 3:
                        idxs = np.random.choice(len(population), 3, replace=False)
                        a, b, c = population[idxs]
                        new_solution = dynamic_crossover(a, b, c)
                    else:
                        new_solution = stochastic_local_exploitation(population[i], scale_factor)
                
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

            population, fitness = update_population()

        return best_solution, best_value