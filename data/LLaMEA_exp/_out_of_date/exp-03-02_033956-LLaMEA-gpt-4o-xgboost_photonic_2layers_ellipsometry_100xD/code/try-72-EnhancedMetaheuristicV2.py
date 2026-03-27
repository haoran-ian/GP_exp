import numpy as np

class EnhancedMetaheuristicV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.8
        self.CR = 0.9
        self.ensemble_factor = 0.2
        self.reshape_probability = 0.3
        self.entropy_threshold = 0.5
        self.phase_transition_factor = 0.12
        self.exploration_intensity = 0.15

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.asarray([func(ind) for ind in population])
        budget_spent = self.population_size

        while budget_spent < self.budget:
            clusters = self.adaptive_clustering(population, fitness)
            region_weights = self.compute_region_weights(population, fitness)
            
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2), lb, ub)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                if np.random.rand() < self.ensemble_factor:
                    nearest_cluster_center = min(clusters, key=lambda c: np.linalg.norm(trial - c))
                    trial += 0.1 * (nearest_cluster_center - trial)

                dynamic_pt_factor = self.phase_transition_factor * (1 - fitness[i] / max(fitness))
                if np.random.rand() < self.reshape_probability:
                    trial += dynamic_pt_factor * np.random.normal(0, 0.1, self.dim)
                
                entropy_measure = -np.sum(np.log(np.abs(fitness - np.mean(fitness)) + 1e-5))
                if entropy_measure < self.entropy_threshold:
                    trial += np.random.normal(0, 0.05, self.dim)

                exploration_factor = self.exploration_intensity * region_weights[i]
                trial += exploration_factor * np.random.normal(0, 0.1, self.dim)

                trial_fitness = func(trial)
                budget_spent += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

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
    
    def compute_region_weights(self, population, fitness):
        min_fitness = np.min(fitness)
        max_fitness = np.max(fitness)
        weights = (max_fitness - fitness) / (max_fitness - min_fitness + 1e-9)
        norm_weights = weights / np.sum(weights)
        return norm_weights