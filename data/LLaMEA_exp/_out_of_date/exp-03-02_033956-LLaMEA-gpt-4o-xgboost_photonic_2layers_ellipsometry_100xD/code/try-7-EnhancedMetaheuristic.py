import numpy as np
from scipy.spatial import distance

class EnhancedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.exploration_factor = 0.15  # Increased exploration factor
        self.phase_selection_probability = 0.2  # Probability of phase-based exploration

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
                    trial += 0.1 * (nearest_cluster_center - trial)
                
                # Stochastic Phase-based Exploration
                if np.random.rand() < self.phase_selection_probability:
                    trial += np.random.normal(0, 0.1, self.dim)
                
                # New Entropy-based Exploration Adjustment
                entropy_measure = -np.sum(np.log(np.abs(fitness - np.mean(fitness)) + 1e-5))
                if entropy_measure < 0.5:  # Adjust threshold as needed
                    trial += np.random.normal(0, 0.05, self.dim)

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
        # Dynamic clustering with reduced cluster size to enhance exploration
        cluster_centers = []
        for i in range(0, self.population_size, max(1, int(self.population_size / 5))):
            cluster_centers.append(np.mean(population[i:i+3], axis=0))
        return np.array(cluster_centers)