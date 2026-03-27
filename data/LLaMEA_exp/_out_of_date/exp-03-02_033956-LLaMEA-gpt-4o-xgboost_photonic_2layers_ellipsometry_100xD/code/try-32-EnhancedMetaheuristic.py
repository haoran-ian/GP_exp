import numpy as np
from scipy.spatial import distance

class EnhancedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.ensemble_factor = 0.2  # Ensemble learning factor
        self.reshape_probability = 0.3  # Probability for dynamic population reshaping
        self.entropy_threshold = 0.5  # Entropy threshold for exploration adjustment
        self.learning_rate = 0.1  # Initial learning rate for adaptive control
        self.memory = []  # Memory for diversity preservation

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.asarray([func(ind) for ind in population])
        budget_spent = self.population_size
        
        while budget_spent < self.budget:
            # Adaptive Clustering for Population Diversity
            clusters = self.adaptive_clustering(population, fitness)

            for i in range(self.population_size):
                # Differential Evolution Mutation with Self-Adaptive Strategy
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                self.F = np.random.uniform(0.5, 1.0)  # Self-Adaptive Differential Weight
                mutant = np.clip(x0 + self.F * (x1 - x2), lb, ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Enhanced Adaptive Local Search using Clustering and Memory
                if np.random.rand() < self.ensemble_factor:
                    nearest_cluster_center = min(clusters, key=lambda c: np.linalg.norm(trial - c))
                    trial += self.learning_rate * (nearest_cluster_center - trial)
                    trial += self.preserve_diversity(trial)

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
                    self.memory.append(trial)  # Update memory for diversity preservation

                if budget_spent >= self.budget:
                    break

            self.update_learning_rate(entropy_measure, fitness)

            # Dynamic Population Reshaping for Exploration
            if np.random.rand() < self.reshape_probability:
                best_indices = np.argsort(fitness)[:self.population_size // 2]
                worst_indices = np.argsort(fitness)[self.population_size // 2:]
                population[worst_indices] = np.random.uniform(lb, ub, (len(worst_indices), self.dim))
                fitness[worst_indices] = [func(ind) for ind in population[worst_indices]]
                budget_spent += len(worst_indices)

        best_index = np.argmin(fitness)
        return population[best_index]

    def adaptive_clustering(self, population, fitness):
        # Adaptive clustering based on fitness to enhance exploration while maintaining diversity
        sorted_indices = np.argsort(fitness)
        cluster_centers = []
        for i in range(0, self.population_size, max(1, int(self.population_size / 5))):
            cluster = population[sorted_indices[i:i+3]]
            if len(cluster) > 0:
                cluster_centers.append(np.mean(cluster, axis=0))
        return np.array(cluster_centers)

    def preserve_diversity(self, trial):
        # Use memory to preserve diversity
        if len(self.memory) > 0:
            diversity_factor = 0.05
            memory_sample = self.memory[np.random.choice(len(self.memory))]
            return diversity_factor * (memory_sample - trial)
        return 0
    
    def update_learning_rate(self, entropy_measure, fitness):
        # Adjust learning rate based on entropy measure and fitness spread
        variance = np.var(fitness)
        if entropy_measure < self.entropy_threshold:
            self.learning_rate = min(0.2, self.learning_rate + 0.01)
        else:
            self.learning_rate = max(0.05, self.learning_rate - 0.01)
        self.learning_rate *= 1 + 0.1 * variance / (1 + variance)  # Variance-based adaptation