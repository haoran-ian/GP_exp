import numpy as np
from sklearn.cluster import KMeans

class EnhancedAdaptiveHybridSearch:
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
            mutation_var = adapt_strength * np.random.rand()
            candidate = x + np.random.randn(self.dim) * mutation_var
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate
        
        def adaptive_mutation(x):
            noise = np.random.randn(self.dim) * (1 - (self.eval_count / self.budget))
            candidate = x + noise
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate

        def differential_crossover(a, b, c):
            F = 0.5 + np.random.rand() * 0.5
            candidate = a + F * (b - c)
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate

        def dynamic_clustering(pop):
            n_clusters = max(2, len(pop) // 10)
            kmeans = KMeans(n_clusters=n_clusters, n_init=1, max_iter=100)
            kmeans.fit(pop)
            return kmeans.labels_

        while self.eval_count < self.budget:
            adapt_strength = 0.5 + 0.5 * (1 - (self.eval_count / self.budget))
            if len(population) > 3:
                labels = dynamic_clustering(population)
            else:
                labels = np.zeros(len(population))

            for i in range(len(population)):
                if self.eval_count >= self.budget:
                    break
                
                cluster_indices = np.where(labels == labels[i])[0]
                if np.random.rand() < 0.5 and len(cluster_indices) > 3:
                    idxs = np.random.choice(cluster_indices, 3, replace=False)
                    a, b, c = population[idxs]
                    new_solution = differential_crossover(a, b, c)
                else:
                    new_solution = local_search(population[i], adapt_strength)
                    
                if memory and np.random.rand() < 0.3:
                    mem_index = np.random.choice(len(memory))
                    new_solution = adaptive_mutation(memory[mem_index])
                
                new_value = func(new_solution)
                self.eval_count += 1
                
                if new_value < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_value
                    memory.append(new_solution)

                    if len(memory) > max_pop_size:
                        memory.pop(0)
                
                if new_value < best_value:
                    best_solution = new_solution
                    best_value = new_value

            if len(population) < max_pop_size and np.random.rand() < 0.2:
                additional = np.random.uniform(bounds[:, 0], bounds[:, 1], (initial_pop_size // 2, self.dim))
                pop_fitness = np.array([func(x) for x in additional])
                self.eval_count += len(additional)
                population = np.vstack((population, additional))
                fitness = np.hstack((fitness, pop_fitness))

        return best_solution, best_value