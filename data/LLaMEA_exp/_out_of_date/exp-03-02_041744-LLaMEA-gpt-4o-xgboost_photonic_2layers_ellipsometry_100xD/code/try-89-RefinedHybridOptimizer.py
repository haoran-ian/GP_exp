import numpy as np
from sklearn.cluster import KMeans

class RefinedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        pop_size = max(10, 5 * self.dim)  
        de_cr = 0.9  
        
        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += pop_size

        def differential_evolution():
            nonlocal population, fitness
            
            for _ in range(self.budget - self.evaluations):
                if self.evaluations >= self.budget:
                    break
                for i in range(pop_size):
                    indices = np.random.choice(pop_size, 3, replace=False)
                    x0, x1, x2 = population[indices]
                    de_f_dynamic = 0.5 + 0.5 * np.random.rand()  
                    mutant = np.clip(x0 + de_f_dynamic * (x1 - x2), bounds[0], bounds[1])
                    cross_points = np.random.rand(self.dim) < de_cr
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[i])
                    f_trial = func(trial)
                    self.evaluations += 1
                    if f_trial < fitness[i]:
                        population[i] = trial
                        fitness[i] = f_trial

        def dynamic_local_search():
            nonlocal population, fitness
            kmeans = KMeans(n_clusters=min(5, pop_size)).fit(population)
            clusters = kmeans.cluster_centers_
            
            for cluster in clusters:
                for _ in range(10):  # Limit local search to a small number of iterations
                    random_step = np.random.uniform(-0.1, 0.1, self.dim)
                    candidate = np.clip(cluster + random_step, bounds[0], bounds[1])
                    candidate_fitness = func(candidate)
                    self.evaluations += 1
                    if candidate_fitness < min(fitness):
                        worst_index = np.argmax(fitness)
                        population[worst_index] = candidate
                        fitness[worst_index] = candidate_fitness

        # Hybrid strategy
        iteration = 0
        while self.evaluations < self.budget:
            if iteration % 2 == 0:
                differential_evolution()
            else:
                dynamic_local_search()
            iteration += 1

        best_index = np.argmin(fitness)
        return population[best_index]