import numpy as np
from scipy.spatial import distance

class RefinedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.9  # Differential weight slightly increased for more aggressive mutations
        self.CR = 0.9  # Crossover probability
        self.exploration_factor = 0.2  # Increased exploration factor
        self.phase_selection_probability = 0.25  # Enhanced probability for phase exploration

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.asarray([func(ind) for ind in population])
        budget_spent = self.population_size
        
        while budget_spent < self.budget:
            # Adaptive Multilevel Clustering for Enhanced Adaptation
            clusters, cluster_probs = self.adaptive_multilevel_clustering(population, fitness)
            
            for i in range(self.population_size):
                # Differential Evolution Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2), lb, ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Enhanced Adaptive Local Search using Multilevel Clustering
                if np.random.rand() < self.exploration_factor:
                    cluster_idx = np.random.choice(len(clusters), p=cluster_probs)
                    trial += 0.1 * (clusters[cluster_idx] - trial)
                
                # Stochastic Phase-based Exploration
                if np.random.rand() < self.phase_selection_probability:
                    trial += np.random.normal(0, 0.1, self.dim)

                # Selection
                trial_fitness = func(trial)
                budget_spent += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                
                if budget_spent >= self.budget:
                    break

        best_index = np.argmin(fitness)
        return population[best_index]
    
    def adaptive_multilevel_clustering(self, population, fitness):
        # Adaptive multilevel clustering with fitness-based probabilities for enhanced exploration
        num_clusters = max(2, int(self.population_size / 4))
        sorted_indices = np.argsort(fitness)
        cluster_centers = [np.mean(population[sorted_indices[i::num_clusters]], axis=0) for i in range(num_clusters)]
        cluster_probs = [1.0 / (i + 1) for i in range(len(cluster_centers))]
        cluster_probs /= np.sum(cluster_probs)
        return np.array(cluster_centers), cluster_probs