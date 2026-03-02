import numpy as np

class AdaptiveMultiPopulationSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.sub_population_size = 10  # Number of individuals per sub-population
        self.num_sub_populations = self.population_size // self.sub_population_size
        self.elite_size = 5
        self.sub_populations = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_values = None
        self.global_best = None
        self.global_best_value = np.inf

    def initialize(self, lb, ub):
        self.sub_populations = [
            np.random.uniform(lb, ub, (self.sub_population_size, self.dim))
            for _ in range(self.num_sub_populations)
        ]
        self.velocities = [
            np.zeros((self.sub_population_size, self.dim))
            for _ in range(self.num_sub_populations)
        ]
        self.personal_best = [np.copy(sp) for sp in self.sub_populations]
        self.personal_best_values = [np.full(self.sub_population_size, np.inf) for _ in range(self.num_sub_populations)]
        self.global_best = np.copy(self.sub_populations[0][0])

    def update_velocity(self, inertia, personal_coefficient, global_coefficient, sp_index):
        r1 = np.random.rand(self.sub_population_size, self.dim)
        r2 = np.random.rand(self.sub_population_size, self.dim)
        cognitive = personal_coefficient * r1 * (self.personal_best[sp_index] - self.sub_populations[sp_index])
        social = global_coefficient * r2 * (self.global_best - self.sub_populations[sp_index])
        self.velocities[sp_index] = inertia * self.velocities[sp_index] + cognitive + social

    def update_position(self, lb, ub, sp_index):
        self.sub_populations[sp_index] += self.velocities[sp_index]
        self.sub_populations[sp_index] = np.clip(self.sub_populations[sp_index], lb, ub)

    def stochastic_hill_climb(self, particle, lb, ub, scale=0.1):
        perturbation = np.random.standard_normal(self.dim) * scale
        neighbor = np.clip(particle + perturbation, lb, ub)
        return neighbor

    def evaluate_sub_population(self, func, sp_index):
        for i in range(self.sub_population_size):
            fitness = func(self.sub_populations[sp_index][i])
            if fitness < self.personal_best_values[sp_index][i]:
                self.personal_best_values[sp_index][i] = fitness
                self.personal_best[sp_index][i] = self.sub_populations[sp_index][i].copy()
            if fitness < self.global_best_value:
                self.global_best_value = fitness
                self.global_best = self.sub_populations[sp_index][i].copy()

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for sp_index in range(self.num_sub_populations):
                inertia = 0.9 - 0.9 * (evaluations / self.budget)
                personal_coefficient = 1.5 + np.random.rand()
                global_coefficient = 1.5 + np.random.rand()

                self.update_velocity(inertia, personal_coefficient, global_coefficient, sp_index)
                self.update_position(lb, ub, sp_index)
                self.evaluate_sub_population(func, sp_index)

                elite_indices = np.argsort(self.personal_best_values[sp_index])[:self.elite_size]
                elites = self.sub_populations[sp_index][elite_indices]

                for i in range(self.sub_population_size):
                    neighbor = self.stochastic_hill_climb(self.sub_populations[sp_index][i], lb, ub)
                    neighbor_fitness = func(neighbor)
                    evaluations += 1
                    if neighbor_fitness < self.personal_best_values[sp_index][i]:
                        self.sub_populations[sp_index][i] = neighbor
                        self.personal_best_values[sp_index][i] = neighbor_fitness
                        self.personal_best[sp_index][i] = neighbor
                    if neighbor_fitness < self.global_best_value:
                        self.global_best_value = neighbor_fitness
                        self.global_best = neighbor

                    if evaluations % 10 == 0:
                        for elite in elites:
                            adaptive_scale = 0.05 * (1 - evaluations / self.budget)
                            enhanced_neighbor = self.stochastic_hill_climb(elite, lb, ub, scale=adaptive_scale)
                            enhanced_fitness = func(enhanced_neighbor)
                            evaluations += 1
                            if enhanced_fitness < self.global_best_value:
                                self.global_best_value = enhanced_fitness
                                self.global_best = enhanced_neighbor

                    if evaluations >= self.budget:
                        break

        return self.global_best_value, self.global_best