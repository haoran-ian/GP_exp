import numpy as np

class EnhancedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.7  # Adaptive F
        self.CR = 0.85  # Adjusted CR
        self.ensemble_factor = 0.25  # Updated ensemble factor
        self.reshape_probability = 0.35  # Adjusted reshape probability
        self.entropy_threshold = 0.4  # Adjusted entropy threshold
        self.phase_transition_factor = 0.15  # Adjusted phase transition factor
        self.learning_rate = 0.1  # Introduced learning rate
        self.elite_fraction = 0.2  # Fraction of best individuals preserved

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

                dynamic_ensemble_factor = self.ensemble_factor * np.std(fitness) / np.mean(fitness)
                if np.random.rand() < dynamic_ensemble_factor:
                    nearest_cluster_center = min(clusters, key=lambda c: np.linalg.norm(trial - c))
                    trial += self.learning_rate * (nearest_cluster_center - trial)

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

            elite_count = int(self.population_size * self.elite_fraction)
            elite_indices = np.argsort(fitness)[:elite_count]
            non_elite_indices = np.argsort(fitness)[elite_count:]
            population[non_elite_indices] = np.random.uniform(lb, ub, (len(non_elite_indices), self.dim))
            fitness[non_elite_indices] = [func(ind) for ind in population[non_elite_indices]]
            budget_spent += len(non_elite_indices)

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