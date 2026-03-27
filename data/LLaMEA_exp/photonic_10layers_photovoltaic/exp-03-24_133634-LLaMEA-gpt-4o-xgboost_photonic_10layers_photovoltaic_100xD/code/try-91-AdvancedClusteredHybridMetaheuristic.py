import numpy as np
from sklearn.cluster import KMeans

class AdvancedClusteredHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * self.dim  # Increased population size for diversity
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Adjusted cooling rate
        self.mutation_factor = 0.85  # Slightly increased mutation factor
        self.crossover_rate = 0.6  # Reduced crossover rate for better exploration
        self.exploration_factor = 0.15  # Increased exploration factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            # Clustering for adaptive local exploration
            kmeans = KMeans(n_clusters=5, random_state=0).fit(population)
            cluster_labels = kmeans.labels_
            diversity = 0

            for i in range(self.population_size):
                # Dynamic mutation based on cluster
                cluster_points = population[cluster_labels == cluster_labels[i]]
                if len(cluster_points) > 1:
                    a, b = np.random.choice(len(cluster_points), 2, replace=False)
                    a, b = cluster_points[a], cluster_points[b]
                else:
                    a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                
                adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                mutant = np.clip(a + adaptive_mutation_factor * (b - population[i]), lb, ub)
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Simulated Annealing
                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness
                    diversity += np.linalg.norm(trial - population[i])

                if budget_used >= self.budget:
                    break

            # Cool down temperature
            self.temperature *= self.cooling_rate

            # Adjust mutation and exploration factors dynamically
            if diversity < 0.15 * (ub - lb).mean():
                self.mutation_factor *= 1.25
                self.exploration_factor *= 1.2

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]