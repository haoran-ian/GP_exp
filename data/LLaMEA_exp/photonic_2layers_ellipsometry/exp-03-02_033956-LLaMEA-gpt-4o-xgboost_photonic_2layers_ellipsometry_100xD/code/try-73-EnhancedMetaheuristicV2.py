import numpy as np

class EnhancedMetaheuristicV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.population_size = self.initial_population_size
        self.F = 0.8
        self.CR = 0.9
        self.ensemble_factor = 0.2
        self.reshape_probability = 0.3
        self.entropy_threshold = 0.5
        self.phase_transition_factor = 0.12
        self.elite_preservation_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.asarray([func(ind) for ind in population])
        budget_spent = self.population_size

        while budget_spent < self.budget:
            clusters = self.adaptive_clustering(population, fitness)
            
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

                trial_fitness = func(trial)
                budget_spent += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_spent >= self.budget:
                    break

            if np.random.rand() < self.reshape_probability:
                elites_count = int(self.elite_preservation_rate * self.population_size)
                best_indices = np.argsort(fitness)[:elites_count]
                worst_indices = np.argsort(fitness)[elites_count:]
                population[worst_indices] = np.random.uniform(lb, ub, (len(worst_indices), self.dim))
                fitness[worst_indices] = [func(ind) for ind in population[worst_indices]]
                budget_spent += len(worst_indices)

            self.dynamic_population_size_adjustment()

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
    
    def dynamic_population_size_adjustment(self):
        self.population_size = max(10, int(self.initial_population_size * (0.5 + np.random.rand())))