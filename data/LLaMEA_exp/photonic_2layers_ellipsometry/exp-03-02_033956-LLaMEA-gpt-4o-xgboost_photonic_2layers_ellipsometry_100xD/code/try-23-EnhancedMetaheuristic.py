import numpy as np
from scipy.spatial import distance

class EnhancedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.8
        self.CR = 0.9
        self.ensemble_factor = 0.2
        self.reshape_probability = 0.3
        self.entropy_threshold = 0.5
        self.inertia_weight = 0.6  # Inertia weight for swarm intelligence
        self.cognitive_coeff = 1.5  # Cognitive coefficient
        self.social_coeff = 1.5  # Social coefficient

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_fitness = np.asarray([func(ind) for ind in personal_best])
        global_best_index = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_index]
        fitness = np.asarray([func(ind) for ind in population])
        budget_spent = self.population_size * 2

        while budget_spent < self.budget:
            clusters = self.adaptive_clustering(population, fitness)
            for i in range(self.population_size):
                # Particle Swarm Optimization Update
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_coeff * r1 * (personal_best[i] - population[i])
                    + self.social_coeff * r2 * (global_best - population[i])
                )
                population[i] = np.clip(population[i] + velocities[i], lb, ub)

                # Adaptive mutation with entropy
                if np.random.rand() < self.ensemble_factor:
                    nearest_cluster_center = min(clusters, key=lambda c: np.linalg.norm(population[i] - c))
                    population[i] += 0.1 * (nearest_cluster_center - population[i])
                entropy_measure = -np.sum(np.log(np.abs(fitness - np.mean(fitness)) + 1e-5))
                if entropy_measure < self.entropy_threshold:
                    population[i] += np.random.normal(0, 0.05, self.dim)

                # Evaluate and update personal and global best
                new_fitness = func(population[i])
                budget_spent += 1
                if new_fitness < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = new_fitness
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness

                if new_fitness < personal_best_fitness[global_best_index]:
                    global_best_index = i
                    global_best = population[i]

                if budget_spent >= self.budget:
                    break

            if np.random.rand() < self.reshape_probability:
                best_indices = np.argsort(fitness)[:self.population_size // 2]
                worst_indices = np.argsort(fitness)[self.population_size // 2:]
                population[worst_indices] = np.random.uniform(lb, ub, (len(worst_indices), self.dim))
                fitness[worst_indices] = [func(ind) for ind in population[worst_indices]]
                budget_spent += len(worst_indices)

        best_index = np.argmin(fitness)
        return population[best_index]

    def adaptive_clustering(self, population, fitness):
        sorted_indices = np.argsort(fitness)
        cluster_centers = []
        for i in range(0, self.population_size, max(1, int(self.population_size / 5))):
            cluster = population[sorted_indices[i:i+3]]
            if len(cluster) > 0:
                cluster_centers.append(np.mean(cluster, axis=0))
        return np.array(cluster_centers)