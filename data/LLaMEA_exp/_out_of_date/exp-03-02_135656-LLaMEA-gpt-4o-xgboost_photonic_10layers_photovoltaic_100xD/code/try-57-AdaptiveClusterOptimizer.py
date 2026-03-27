import numpy as np

class AdaptiveClusterOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.7  # Fine-tuned cognitive component
        self.c2 = 2.3  # Fine-tuned social component
        self.inertia_weight = 0.9
        self.velocity_clamp = 0.1
        self.cluster_radius = 0.15
        self.mutation_probability = 0.1
        self.population = None
        self.velocities = None
        self.personal_best = None
        self.global_best = None
        self.fitness_evals = 0

    def initialize(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-self.velocity_clamp, self.velocity_clamp, (self.population_size, self.dim))
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_fitness = np.inf

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness_evals >= self.budget:
                break
            fitness = func(self.population[i])
            self.fitness_evals += 1
            if fitness < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness
                self.personal_best[i] = self.population[i].copy()
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best = self.population[i].copy()

    def update_velocities_and_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.inertia_weight = 0.5 + (0.4 * np.cos(np.pi * self.fitness_evals / self.budget))
        for i in range(self.population_size):
            inertia = self.inertia_weight * self.velocities[i]
            cognitive = self.c1 * np.random.rand(self.dim) * (self.personal_best[i] - self.population[i])
            social = self.c2 * np.random.rand(self.dim) * (self.global_best - self.population[i])
            cluster_influence = self.cluster_guided_exploration(i)
            self.velocities[i] = inertia + cognitive + social + 0.2 * cluster_influence
            self.velocity_clamp = 0.1 * (1 - (self.fitness_evals / self.budget))
            self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)
            self.population[i] += self.velocities[i]
            self.population[i] = self.mutate(self.population[i], lb, ub)
            self.population[i] = np.clip(self.population[i], lb, ub)

    def cluster_guided_exploration(self, i):
        dist = np.linalg.norm(self.population - self.population[i], axis=1)
        cluster_indices = np.where(dist < self.cluster_radius)[0]
        if cluster_indices.size > 1:
            cluster_center = np.mean(self.population[cluster_indices], axis=0)
            return cluster_center - self.population[i]
        return np.zeros(self.dim)

    def mutate(self, individual, lb, ub):
        dynamic_mutation_probability = self.mutation_probability * (1 - (self.fitness_evals / self.budget))
        if np.random.rand() < dynamic_mutation_probability:
            mutation_vector = np.random.normal(0, 0.1, self.dim)
            individual += mutation_vector
        return individual

    def __call__(self, func):
        bounds = func.bounds
        self.initialize(bounds)
        while self.fitness_evals < self.budget:
            self.evaluate_population(func)
            self.update_velocities_and_positions(bounds)
        return self.global_best