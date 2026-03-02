import numpy as np

class DynamicDiversitySwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_size = 5
        self.swarm = None
        self.velocity = None
        self.personal_best = None
        self.personal_best_values = None
        self.global_best = None
        self.global_best_value = np.inf

    def initialize(self, lb, ub):
        self.swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocity = np.zeros((self.population_size, self.dim))
        self.personal_best = np.copy(self.swarm)
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best = np.copy(self.swarm[0])

    def update_velocity(self, inertia, personal_coefficient, global_coefficient):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        cognitive = personal_coefficient * r1 * (self.personal_best - self.swarm)
        social = global_coefficient * r2 * (self.global_best - self.swarm)
        self.velocity = inertia * self.velocity + cognitive + social

    def update_position(self, lb, ub):
        self.swarm += self.velocity
        self.swarm = np.clip(self.swarm, lb, ub)

    def adaptive_neighborhood_search(self, particle, fitness, lb, ub, scale=0.1):
        perturbation = np.random.standard_normal(self.dim) * scale
        neighbor = np.clip(particle + perturbation, lb, ub)
        neighbor_fitness = func(neighbor)
        if neighbor_fitness < fitness:
            return neighbor, neighbor_fitness
        return particle, fitness

    def evaluate_population(self, func):
        for i in range(self.population_size):
            fitness = func(self.swarm[i])
            if fitness < self.personal_best_values[i]:
                self.personal_best_values[i] = fitness
                self.personal_best[i] = self.swarm[i].copy()
            if fitness < self.global_best_value:
                self.global_best_value = fitness
                self.global_best = self.swarm[i].copy()

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            inertia = 0.9 - 0.7 * (evaluations / self.budget)
            personal_coefficient = 1.5 + np.random.uniform(-0.5, 0.5)
            global_coefficient = 1.5 + np.random.uniform(-0.5, 0.5)

            self.update_velocity(inertia, personal_coefficient, global_coefficient)
            self.update_position(lb, ub)
            self.evaluate_population(func)
            
            swarm_diversity = np.std(self.swarm, axis=0).mean()
            adaptive_scale = 0.1 + 0.4 * (1 - (swarm_diversity / (ub - lb).mean()))

            for i in range(self.population_size):
                self.swarm[i], self.personal_best_values[i] = self.adaptive_neighborhood_search(
                    self.swarm[i], self.personal_best_values[i], lb, ub, scale=adaptive_scale)
                
                evaluations += 1

                if self.personal_best_values[i] < self.global_best_value:
                    self.global_best_value = self.personal_best_values[i]
                    self.global_best = self.swarm[i].copy()
                
                if evaluations >= self.budget:
                    break

        return self.global_best_value, self.global_best