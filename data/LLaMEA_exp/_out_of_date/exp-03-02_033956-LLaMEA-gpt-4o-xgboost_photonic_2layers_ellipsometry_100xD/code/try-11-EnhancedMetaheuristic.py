import numpy as np

class EnhancedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.cluster_refresh_rate = 10  # Determines when to refresh clusters
        self.entropy_threshold = 0.5  # Entropy threshold for adjustments
        self.entropy_decay = 0.95  # Decay factor for adaptive entropy threshold

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.asarray([func(ind) for ind in population])
        budget_spent = self.population_size
        cluster_centers = self.dynamic_clustering(population, fitness)

        iteration = 0
        while budget_spent < self.budget:
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

                # Multi-phase Entrainment
                if np.random.rand() < 0.2:
                    trial += np.random.uniform(-0.1, 0.1, self.dim)

                # Adaptive Entropy-based Exploration
                entropy_measure = -np.sum(np.log(np.abs(fitness - np.mean(fitness)) + 1e-5))
                if entropy_measure < self.entropy_threshold:
                    trial += np.random.normal(0, 0.05, self.dim)
                    self.entropy_threshold *= self.entropy_decay

                # Selection
                trial_fitness = func(trial)
                budget_spent += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_spent >= self.budget:
                    break

            # Refresh clusters periodically
            if iteration % self.cluster_refresh_rate == 0:
                cluster_centers = self.dynamic_clustering(population, fitness)
            iteration += 1

        best_index = np.argmin(fitness)
        return population[best_index]

    def dynamic_clustering(self, population, fitness):
        # Dynamic clustering based on fitness to facilitate adaptive exploration
        sorted_indices = np.argsort(fitness)
        clusters = []
        for i in range(0, self.population_size, max(1, int(self.population_size / 5))):
            cluster = population[sorted_indices[i:i + 3]]
            if len(cluster) > 0:
                clusters.append(np.mean(cluster, axis=0))
        return np.array(clusters)