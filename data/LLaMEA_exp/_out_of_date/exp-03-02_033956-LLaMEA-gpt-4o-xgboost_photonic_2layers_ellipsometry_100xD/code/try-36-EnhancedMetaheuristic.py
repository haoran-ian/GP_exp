import numpy as np
from scipy.spatial import distance

class EnhancedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F_lower, self.F_upper = 0.5, 0.9  # Self-adaptive differential weight range
        self.CR_lower, self.CR_upper = 0.3, 0.9  # Self-adaptive crossover probability range
        self.ensemble_factor = 0.2  # Ensemble learning factor
        self.reshape_probability = 0.3  # Probability for dynamic population reshaping
        self.entropy_threshold = 0.5  # Entropy threshold for exploration adjustment

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.chaotic_initialization(lb, ub, self.population_size, self.dim)
        fitness = np.asarray([func(ind) for ind in population])
        budget_spent = self.population_size
        
        while budget_spent < self.budget:
            # Adaptive Clustering for Population Diversity
            clusters = self.adaptive_clustering(population, fitness)
            
            for i in range(self.population_size):
                # Self-adaptive Differential Evolution Mutation
                F = np.random.uniform(self.F_lower, self.F_upper)
                CR = np.random.uniform(self.CR_lower, self.CR_upper)
                
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + F * (x1 - x2), lb, ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Enhanced Adaptive Local Search using Clustering
                if np.random.rand() < self.ensemble_factor:
                    nearest_cluster_center = min(clusters, key=lambda c: np.linalg.norm(trial - c))
                    trial += 0.1 * (nearest_cluster_center - trial)
                
                # Stochastic Phase-based Exploration
                if np.random.rand() < self.reshape_probability:
                    trial += np.random.normal(0, 0.1, self.dim)
                
                # New Entropy-based Exploration Adjustment with Feedback
                entropy_measure = -np.sum(np.log(np.abs(fitness - np.mean(fitness)) + 1e-5))
                if entropy_measure < self.entropy_threshold:
                    trial += np.random.normal(0, 0.05, self.dim)

                # Selection
                trial_fitness = func(trial)
                budget_spent += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                
                if budget_spent >= self.budget:
                    break
            
            # Dynamic Population Reshaping for Exploration
            if np.random.rand() < self.reshape_probability:
                best_indices = np.argsort(fitness)[:self.population_size // 2]
                worst_indices = np.argsort(fitness)[self.population_size // 2:]
                population[worst_indices] = self.chaotic_initialization(lb, ub, len(worst_indices), self.dim)
                fitness[worst_indices] = [func(ind) for ind in population[worst_indices]]
                budget_spent += len(worst_indices)

        best_index = np.argmin(fitness)
        return population[best_index]
    
    def chaotic_initialization(self, lb, ub, size, dim):
        # Chaotic map initialization for enhanced exploration
        a = 0.7  # Logistic map parameter
        x = np.random.rand(size, dim)
        for _ in range(100):  # Iterate chaotic map
            x = a * x * (1 - x)
        return lb + (ub - lb) * x
    
    def adaptive_clustering(self, population, fitness):
        # Adaptive clustering based on fitness to enhance exploration while maintaining diversity
        sorted_indices = np.argsort(fitness)
        cluster_centers = []
        for i in range(0, self.population_size, max(1, int(self.population_size / 5))):
            cluster = population[sorted_indices[i:i+3]]
            if len(cluster) > 0:
                cluster_centers.append(np.mean(cluster, axis=0))
        return np.array(cluster_centers)