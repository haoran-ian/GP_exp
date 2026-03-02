import numpy as np

class RefinedHybridSwarmEvolution:
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
        # Chaotic initialization using Logistic Map for better coverage
        x = np.random.rand(self.population_size, self.dim)
        r = 4.0  # Chaotic parameter
        self.swarm = lb + (ub - lb) * (r * x * (1 - x))
        self.velocity = np.random.standard_normal((self.population_size, self.dim))
        self.personal_best = np.copy(self.swarm)
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best = np.copy(self.swarm[0])

    def update_velocity(self, evaluations, inertia_phase):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        if inertia_phase == "exploration":
            inertia = 0.9
        else:
            inertia = 0.4
        personal_coefficient = 2.0 * np.random.rand()
        global_coefficient = 2.0 * np.random.rand()
        cognitive = personal_coefficient * r1 * (self.personal_best - self.swarm)
        social = global_coefficient * r2 * (self.global_best - self.swarm)
        self.velocity = inertia * self.velocity + cognitive + social

    def update_position(self, lb, ub):
        self.swarm += self.velocity
        self.swarm = np.clip(self.swarm, lb, ub)

    def stochastic_hill_climb(self, particle, lb, ub, scale=0.1):
        perturbation = np.random.standard_normal(self.dim) * scale
        neighbor = np.clip(particle + perturbation, lb, ub)
        return neighbor

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
            inertia_phase = "exploration" if evaluations < 0.5 * self.budget else "exploitation"
            self.update_velocity(evaluations, inertia_phase)
            self.update_position(lb, ub)
            self.evaluate_population(func)

            elite_indices = np.argsort(self.personal_best_values)[:self.elite_size]
            elites = self.swarm[elite_indices]

            for i in range(self.population_size):
                neighbor = self.stochastic_hill_climb(self.swarm[i], lb, ub)
                neighbor_fitness = func(neighbor)
                evaluations += 1
                if neighbor_fitness < self.personal_best_values[i]:
                    self.swarm[i] = neighbor
                    self.personal_best_values[i] = neighbor_fitness
                    self.personal_best[i] = neighbor
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