import numpy as np

class EnhancedRefinedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 25
        self.F = 0.9  # Differential weight
        self.CR = 0.8  # Crossover probability
        self.exploration_factor = 0.2  # Exploration factor
        self.phase_selection_probability = 0.25  # Phase-based exploration probability
        self.entropy_threshold = 0.4  # Entropy threshold for exploration adjustment

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.asarray([func(ind) for ind in population])
        budget_spent = self.population_size
        
        while budget_spent < self.budget:
            # Dynamic Clustering for Improved Diversity Control
            clusters = self.dynamic_clustering(population, fitness)
            
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
                
                # Enhanced Adaptive Local Search with Dynamic Clustering
                if np.random.rand() < self.exploration_factor:
                    nearest_cluster_center = min(clusters, key=lambda c: np.linalg.norm(trial - c))
                    trial += 0.15 * (nearest_cluster_center - trial)
                
                # Expanded Phase-based Exploration
                if np.random.rand() < self.phase_selection_probability:
                    trial += np.random.normal(0, 0.15, self.dim)
                
                # Entropy-controlled Diversification
                entropy_measure = -np.sum(np.log(np.abs(fitness - np.mean(fitness)) + 1e-5))
                if entropy_measure < self.entropy_threshold:  # Adjust threshold as needed
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
    
    def dynamic_clustering(self, population, fitness):
        # Dynamic clustering to balance exploration and exploitation dynamically
        sorted_indices = np.argsort(fitness)
        cluster_centers = []
        cluster_size = max(1, int(self.population_size / 4))
        for i in range(0, self.population_size, cluster_size):
            cluster = population[sorted_indices[i:i+cluster_size]]
            if len(cluster) > 0:
                cluster_centers.append(np.mean(cluster, axis=0))
        return np.array(cluster_centers)