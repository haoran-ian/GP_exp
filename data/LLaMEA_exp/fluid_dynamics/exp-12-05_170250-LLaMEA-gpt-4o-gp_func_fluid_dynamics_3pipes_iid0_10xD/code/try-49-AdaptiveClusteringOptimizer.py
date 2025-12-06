import numpy as np

class AdaptiveClusteringOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(100, budget)  # Initial population size
        self.cluster_count = max(2, self.dim // 3)  # Adjusted number of clusters

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
    
    def evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def cluster_population(self, population, scores):
        sorted_indices = np.argsort(scores)
        sorted_population = population[sorted_indices]
        clusters = np.array_split(sorted_population, self.cluster_count)
        return clusters

    def explore_exploit(self, cluster):
        cluster_center = np.mean(cluster, axis=0)
        perturbation = np.random.uniform(-0.4, 0.4, cluster.shape)  # Expanded perturbation range
        scaling_factor = 0.5 + np.std(cluster, axis=0) / 8  # Refined scaling factor adjustment
        mutation_factor = np.random.normal(0, 0.25, cluster.shape)  # Refine mutation factor
        new_candidates = cluster_center + (perturbation + mutation_factor) * scaling_factor
        new_candidates = np.clip(new_candidates, self.lower_bound, self.upper_bound)
        return new_candidates

    def __call__(self, func):
        population = self.initialize_population()
        evaluations = 0

        while evaluations + self.population_size <= self.budget:
            scores = self.evaluate_population(population, func)
            evaluations += self.population_size
            
            clusters = self.cluster_population(population, scores)
            new_population = []

            for cluster in clusters:
                new_candidates = self.explore_exploit(cluster)
                new_population.extend(new_candidates)

            population = np.array(new_population[:self.population_size])
        
        # Final evaluation of the remaining budget
        if evaluations < self.budget:
            remaining_budget = self.budget - evaluations
            scores = self.evaluate_population(population[:remaining_budget], func)
        
        best_index = np.argmin(self.evaluate_population(population, func))
        return population[best_index]