import numpy as np

class EnhancedSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0  # cognitive component
        self.c2 = 2.0  # social component
        self.inertia_weight = 0.9  # Adjusted initial inertia weight
        self.velocity_clamp = 0.1
        self.local_search_intensity = 0.1
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
        elite = self.select_elite_members()
        for i in range(self.population_size):
            inertia = self.inertia_weight * self.velocities[i]
            cognitive = self.c1 * np.random.rand(self.dim) * (self.personal_best[i] - self.population[i])
            social = self.c2 * np.random.rand(self.dim) * (self.global_best - self.population[i])
            elite_influence = np.mean(elite, axis=0) - self.population[i]
            self.velocities[i] = inertia + cognitive + social + 0.5 * elite_influence
            self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)
            self.population[i] += self.velocities[i] * self.local_search_adaptive_factor(i)
            self.population[i] = self.mutate(self.population[i], lb, ub)
            self.population[i] = np.clip(self.population[i], lb, ub)
            # Decrease inertia weight linearly
            self.inertia_weight *= 0.99

    def local_search_adaptive_factor(self, i):
        return self.local_search_intensity * (1 - (self.personal_best_fitness[i] / self.global_best_fitness))

    def select_elite_members(self):
        elite_size = max(1, int(0.1 * self.population_size))
        elite_indices = np.argsort(self.personal_best_fitness)[:elite_size]
        return self.population[elite_indices]

    def mutate(self, individual, lb, ub):
        if np.random.rand() < self.mutation_probability:
            mutation_vector = np.random.normal(0, 0.1, self.dim)
            individual += mutation_vector
        return individual

    def __call__(self, func):
        bounds = func.bounds
        self.initialize(bounds)
        while self.fitness_evals < self.budget:
            self.evaluate_population(func)
            self.update_velocities_and_positions(bounds)
        return self.global_best  # Return the best solution found