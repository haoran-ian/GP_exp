import numpy as np
from sklearn.cluster import KMeans

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
        max_pop_size = initial_pop_size * 5
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (initial_pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.eval_count += initial_pop_size
        
        memory = []

        def local_search(x, adapt_strength):
            mutation_var = adapt_strength * np.random.randn(self.dim)
            candidate = x + mutation_var
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate
        
        def hybrid_mutation(x, adapt_strength):
            noise = np.random.randn(self.dim) * adapt_strength
            candidate = x + noise
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate

        def adaptive_crossover(a, b, c):
            F = 0.4 + 0.6 * np.random.rand() * (1 - (self.eval_count / self.budget))
            candidate = a + F * (b - c)
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate

        def adaptive_clustering():
            if len(memory) > 5:
                kmeans = KMeans(n_clusters=min(len(memory) // 5, 5), random_state=0).fit(memory)
                cluster_centers = kmeans.cluster_centers_
                for center in cluster_centers:
                    if self.eval_count >= self.budget:
                        break
                    new_solution = hybrid_mutation(center, adapt_strength)
                    new_value = func(new_solution)
                    self.eval_count += 1
                    if new_value < best_value:
                        nonlocal best_solution, best_value
                        best_solution, best_value = new_solution, new_value

        while self.eval_count < self.budget:
            adapt_strength = 0.5 * (1 - (self.eval_count / self.budget))
            
            for i in range(len(population)):
                if self.eval_count >= self.budget:
                    break
                
                if np.random.rand() < 0.3:
                    new_solution = np.random.uniform(bounds[:, 0], bounds[:, 1], self.dim)
                else:
                    if memory and np.random.rand() < 0.4:
                        mem_index = np.random.choice(len(memory))
                        new_solution = hybrid_mutation(memory[mem_index], adapt_strength)
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

            adaptive_clustering()

            if len(population) < max_pop_size and np.random.rand() < 0.2:
                additional = np.random.uniform(bounds[:, 0], bounds[:, 1], (initial_pop_size // 2, self.dim))
                pop_fitness = np.array([func(x) for x in additional])
                self.eval_count += len(additional)
                population = np.vstack((population, additional))
                fitness = np.hstack((fitness, pop_fitness))

        return best_solution, best_value