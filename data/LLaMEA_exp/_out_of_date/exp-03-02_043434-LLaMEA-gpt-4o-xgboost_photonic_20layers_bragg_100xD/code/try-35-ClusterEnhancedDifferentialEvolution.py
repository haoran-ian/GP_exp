import numpy as np
from sklearn.cluster import KMeans

class ClusterEnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 * dim
        self.population = np.random.rand(self.population_size, dim)
        self.initial_population_size = self.population_size
        self.F_base = 0.5
        self.CR_base = 0.9
        self.inertia_weight = 0.9
        self.cluster_update_freq = 10

    def levy_flight(self, L, scale):
        u = np.random.normal(0, 1, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = scale * u / np.abs(v) ** (1 / L)
        return step

    def chaotic_local_search(self, position, lb, ub, chaos_level=0.1):
        chaotic_step = chaos_level * (np.random.rand(self.dim) - 0.5) * (ub - lb)
        new_position = position + chaotic_step
        return np.clip(new_position, lb, ub)

    def update_population_clusters(self):
        kmeans = KMeans(n_clusters=max(2, self.population_size // 10))
        kmeans.fit(self.population)
        return kmeans.labels_

    def differential_evolution(self, func, lb, ub):
        bounds = np.array([lb, ub])
        best_solution = None
        best_fitness = np.inf
        evaluations = 0
        cluster_labels = self.update_population_clusters()

        self.population = lb + (ub - lb) * self.population
        fitness = np.apply_along_axis(func, 1, self.population)

        while evaluations < self.budget:
            self.population_size = max(int(self.initial_population_size * (1 - evaluations / self.budget)), 4)
            
            if evaluations % self.cluster_update_freq == 0:
                cluster_labels = self.update_population_clusters()

            for i in range(self.population_size):
                cluster_members = self.population[cluster_labels == cluster_labels[i]]
                if len(cluster_members) >= 3:
                    a, b, c = cluster_members[np.random.choice(len(cluster_members), 3, replace=False)]
                else:
                    indices = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = self.population[np.random.choice(indices, 3, replace=False)]

                F_dynamic = self.F_base + self.inertia_weight * np.random.rand()
                CR_dynamic = self.CR_base - self.inertia_weight * np.random.rand()

                mutant_vector = np.clip(a + F_dynamic * (b - c), lb, ub)
                crossover_mask = np.random.rand(self.dim) < CR_dynamic
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])

                levy_scale = 0.1 + 0.4 * (1 - evaluations / self.budget)
                trial_vector += self.levy_flight(1.5, levy_scale) * (trial_vector - self.population[i])

                trial_vector = self.chaotic_local_search(trial_vector, lb, ub)

                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    self.population[i] = trial_vector
                    fitness[i] = trial_fitness

                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial_vector

                if evaluations >= self.budget:
                    break

        return best_solution, best_fitness

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution, best_fitness = self.differential_evolution(func, lb, ub)
        return best_solution