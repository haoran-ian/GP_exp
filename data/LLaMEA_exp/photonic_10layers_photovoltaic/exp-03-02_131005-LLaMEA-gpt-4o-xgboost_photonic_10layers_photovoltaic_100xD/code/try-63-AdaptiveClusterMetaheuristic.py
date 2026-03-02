import numpy as np
from sklearn.cluster import DBSCAN

class AdaptiveClusterMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Define search space boundaries
        lb, ub = func.bounds.lb, func.bounds.ub

        # Initialize population
        population_size = int(np.sqrt(self.budget))
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([self._evaluate(func, ind) for ind in population])

        # Dynamic strategy parameters
        exploration_factor = 0.5
        exploitation_factor = 0.5
        mutation_rate = 0.1

        while self.evaluations < self.budget:
            # Density-based clustering for local structure exploitation
            cluster_labels = self._density_clustering(population, fitness)
            cluster_centers = self._compute_cluster_centers(population, cluster_labels, fitness)

            # Generate offspring using exploration and exploitation
            for i in range(population_size):
                if np.random.rand() < exploration_factor:
                    # Exploration: Adaptive chaotic perturbation
                    trial = self._chaotic_random_perturbation(population[i], lb, ub, mutation_rate)
                else:
                    # Exploitation: Density-based guided local search
                    trial = self._density_guided_local_search(population[i], cluster_centers, lb, ub)

                trial_fitness = self._evaluate(func, trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Adaptive phase adjustment based on cluster evaluations
            if self._phase_transition_condition(fitness):
                exploration_factor *= 0.85  
                exploitation_factor *= 1.15 
                mutation_rate *= 0.95  

            # Adjust mutation rate based on variance in fitness
            mutation_rate *= (1 - 0.4 * np.var(fitness) / np.mean(fitness))

            if self.evaluations >= self.budget:
                break

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def _evaluate(self, func, individual):
        if self.evaluations >= self.budget:
            raise RuntimeError("Exceeded budget")
        self.evaluations += 1
        return func(individual)

    def _chaotic_random_perturbation(self, individual, lb, ub, mutation_rate):
        chaos_factor = np.random.normal(0, mutation_rate, size=self.dim)
        perturbation = np.sin(chaos_factor) * mutation_rate
        trial = np.clip(individual + perturbation, lb, ub)
        return trial

    def _density_guided_local_search(self, individual, cluster_centers, lb, ub):
        closest_center = min(cluster_centers, key=lambda center: np.linalg.norm(center - individual))
        direction = closest_center - individual
        chaos_direction = np.sin(direction) * 0.1
        trial = np.clip(individual + chaos_direction, lb, ub)
        return trial

    def _density_clustering(self, population, fitness):
        clustering_algo = DBSCAN(eps=0.2, min_samples=2)
        clustering_algo.fit(population)
        return clustering_algo.labels_

    def _compute_cluster_centers(self, population, cluster_labels, fitness):
        unique_labels = set(cluster_labels)
        cluster_centers = []
        for label in unique_labels:
            if label != -1:
                members = population[cluster_labels == label]
                member_fitness = fitness[cluster_labels == label]
                center = members[np.argmin(member_fitness)]
                cluster_centers.append(center)
        return cluster_centers

    def _phase_transition_condition(self, fitness):
        sorted_fitness = np.sort(fitness)
        phase_threshold = np.percentile(sorted_fitness, 15)
        return np.any(sorted_fitness[:int(0.15 * len(fitness))] < phase_threshold)