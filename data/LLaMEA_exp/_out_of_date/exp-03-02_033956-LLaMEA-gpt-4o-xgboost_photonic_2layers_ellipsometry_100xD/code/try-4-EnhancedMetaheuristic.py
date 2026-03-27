import numpy as np
from scipy.spatial import distance

class EnhancedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 25  # Increased population size
        self.F = 0.9  # Adjusted differential weight
        self.CR = 0.85  # Adjusted crossover probability
        self.exploration_factor = 0.2  # Increased exploration factor
        self.phase_selection_probability = 0.25  # Increased phase-based exploration probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.asarray([func(ind) for ind in population])
        budget_spent = self.population_size
        
        while budget_spent < self.budget:
            # Dynamic Clustering for Adaptation
            clusters = self.dynamic_clustering(population)
            
            for i in range(self.population_size):
                # Differential Evolution Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:  # Ensure current index is not in selected indices
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2), lb, ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Enhanced Adaptive Local Search using Clustering
                if np.random.rand() < self.exploration_factor:
                    nearest_cluster_center = min(clusters, key=lambda c: np.linalg.norm(trial - c))
                    trial += 0.15 * (nearest_cluster_center - trial)  # Adjusted exploration step
                
                # Stochastic Phase-based Exploration
                if np.random.rand() < self.phase_selection_probability:
                    trial += np.random.normal(0, 0.2, self.dim)  # Adjusted exploration variance

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
    
    def dynamic_clustering(self, population):
        # Hierarchical clustering with adaptive cluster size
        cluster_centers = []
        for i in range(0, self.population_size, max(1, int(self.population_size / 4))):  # Adjusted cluster size
            cluster_centers.append(np.mean(population[i:i+4], axis=0))
        return np.array(cluster_centers)