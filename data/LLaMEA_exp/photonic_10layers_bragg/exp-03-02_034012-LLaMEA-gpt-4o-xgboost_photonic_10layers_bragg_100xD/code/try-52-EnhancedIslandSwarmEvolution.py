import numpy as np

class EnhancedIslandSwarmEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_size = 5
        self.num_islands = 3
        self.islands = [None] * self.num_islands
        self.velocities = [None] * self.num_islands
        self.personal_bests = [None] * self.num_islands
        self.personal_best_values = [None] * self.num_islands
        self.global_best = None
        self.global_best_value = np.inf

    def initialize(self, lb, ub):
        for i in range(self.num_islands):
            self.islands[i] = np.random.uniform(lb, ub, (self.population_size, self.dim))
            self.velocities[i] = np.zeros((self.population_size, self.dim))
            self.personal_bests[i] = np.copy(self.islands[i])
            self.personal_best_values[i] = np.full(self.population_size, np.inf)

    def update_velocity(self, inertia, personal_coefficient, global_coefficient, island_idx):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        cognitive = personal_coefficient * r1 * (self.personal_bests[island_idx] - self.islands[island_idx])
        social = global_coefficient * r2 * (self.global_best - self.islands[island_idx])
        self.velocities[island_idx] = inertia * self.velocities[island_idx] + cognitive + social

    def update_position(self, lb, ub, island_idx):
        self.islands[island_idx] += self.velocities[island_idx]
        self.islands[island_idx] = np.clip(self.islands[island_idx], lb, ub)

    def stochastic_hill_climb(self, particle, lb, ub, scale=0.1):
        perturbation = np.random.standard_normal(self.dim) * scale
        neighbor = np.clip(particle + perturbation, lb, ub)
        return neighbor

    def evaluate_population(self, func, island_idx):
        for i in range(self.population_size):
            fitness = func(self.islands[island_idx][i])
            if fitness < self.personal_best_values[island_idx][i]:
                self.personal_best_values[island_idx][i] = fitness
                self.personal_bests[island_idx][i] = self.islands[island_idx][i].copy()
            if fitness < self.global_best_value:
                self.global_best_value = fitness
                self.global_best = self.islands[island_idx][i].copy()

    def migrate(self):
        migrants = [self.islands[i][np.random.choice(self.population_size, self.elite_size, replace=False)] for i in range(self.num_islands)]
        for i in range(self.num_islands):
            for j in range(self.elite_size):
                src_island = (i + 1) % self.num_islands
                self.islands[i][np.random.randint(self.population_size)] = migrants[src_island][j]

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for island_idx in range(self.num_islands):
                inertia = 0.9 - 0.9 * (evaluations / self.budget)
                personal_coefficient = 1.5 + np.random.rand()
                global_coefficient = 1.5 + np.random.rand()

                self.update_velocity(inertia, personal_coefficient, global_coefficient, island_idx)
                self.update_position(lb, ub, island_idx)
                self.evaluate_population(func, island_idx)

                for i in range(self.population_size):
                    neighbor = self.stochastic_hill_climb(self.islands[island_idx][i], lb, ub)
                    neighbor_fitness = func(neighbor)
                    evaluations += 1
                    if neighbor_fitness < self.personal_best_values[island_idx][i]:
                        self.islands[island_idx][i] = neighbor
                        self.personal_best_values[island_idx][i] = neighbor_fitness
                        self.personal_bests[island_idx][i] = neighbor
                    if neighbor_fitness < self.global_best_value:
                        self.global_best_value = neighbor_fitness
                        self.global_best = neighbor

                if evaluations % 100 == 0:
                    self.migrate()
                
                if evaluations >= self.budget:
                    break

        return self.global_best_value, self.global_best