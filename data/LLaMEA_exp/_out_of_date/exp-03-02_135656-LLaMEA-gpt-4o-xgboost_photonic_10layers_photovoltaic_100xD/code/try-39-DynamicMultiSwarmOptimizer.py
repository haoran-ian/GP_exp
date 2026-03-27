import numpy as np

class DynamicMultiSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.num_swarms = 5
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_weight = 0.9
        self.velocity_clamp = 0.1
        self.local_search_intensity = 0.1
        self.mutation_probability = 0.1
        self.populations = [None] * self.num_swarms
        self.velocities = [None] * self.num_swarms
        self.personal_best = [None] * self.num_swarms
        self.global_best = [None] * self.num_swarms
        self.fitness_evals = 0

    def initialize(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        for s in range(self.num_swarms):
            self.populations[s] = np.random.uniform(lb, ub, (self.population_size // self.num_swarms, self.dim))
            self.velocities[s] = np.random.uniform(-self.velocity_clamp, self.velocity_clamp, (self.population_size // self.num_swarms, self.dim))
            self.personal_best[s] = self.populations[s].copy()
            self.personal_best_fitness = [np.full(self.population_size // self.num_swarms, np.inf) for _ in range(self.num_swarms)]
            self.global_best[s] = None
            self.global_best_fitness = np.inf

    def evaluate_population(self, func, swarm_idx):
        for i in range(self.population_size // self.num_swarms):
            if self.fitness_evals >= self.budget:
                break
            fitness = func(self.populations[swarm_idx][i])
            self.fitness_evals += 1
            if fitness < self.personal_best_fitness[swarm_idx][i]:
                self.personal_best_fitness[swarm_idx][i] = fitness
                self.personal_best[swarm_idx][i] = self.populations[swarm_idx][i].copy()
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best[swarm_idx] = self.populations[swarm_idx][i].copy()

    def update_velocities_and_positions(self, bounds, swarm_idx):
        lb, ub = bounds.lb, bounds.ub
        self.inertia_weight = 0.4 + (0.5 * (self.budget - self.fitness_evals) / self.budget)
        for i in range(self.population_size // self.num_swarms):
            inertia = self.inertia_weight * self.velocities[swarm_idx][i]
            cognitive = self.c1 * np.random.rand(self.dim) * (self.personal_best[swarm_idx][i] - self.populations[swarm_idx][i])
            social = self.c2 * np.random.rand(self.dim) * (self.global_best[swarm_idx] - self.populations[swarm_idx][i])
            inter_swarm_influence = self.inter_swarm_influence(swarm_idx, i)
            self.velocities[swarm_idx][i] = inertia + cognitive + social + inter_swarm_influence
            self.velocity_clamp = 0.1 * (1 - (self.fitness_evals / self.budget))
            self.velocities[swarm_idx][i] = np.clip(self.velocities[swarm_idx][i], -self.velocity_clamp, self.velocity_clamp)
            self.populations[swarm_idx][i] += self.velocities[swarm_idx][i] * self.local_search_adaptive_factor(swarm_idx, i)
            self.populations[swarm_idx][i] = self.mutate(self.populations[swarm_idx][i], lb, ub)
            self.populations[swarm_idx][i] = np.clip(self.populations[swarm_idx][i], lb, ub)

    def local_search_adaptive_factor(self, swarm_idx, i):
        return self.local_search_intensity * (1 - (self.personal_best_fitness[swarm_idx][i] / self.global_best_fitness))

    def inter_swarm_influence(self, current_swarm, i):
        best_other_swarm = None
        best_other_fitness = np.inf
        for s in range(self.num_swarms):
            if s != current_swarm and self.global_best_fitness < best_other_fitness:
                best_other_fitness = self.global_best_fitness
                best_other_swarm = s
        if best_other_swarm is not None:
            influence = np.random.rand(self.dim) * (self.global_best[best_other_swarm] - self.populations[current_swarm][i])
            return influence
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
            for s in range(self.num_swarms):
                self.evaluate_population(func, s)
                self.update_velocities_and_positions(bounds, s)
        best_swarm_idx = np.argmin([self.global_best_fitness for _ in range(self.num_swarms)])
        return self.global_best[best_swarm_idx]