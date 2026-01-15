import numpy as np

class ChaoticHybridQSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(30, budget // 5)
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.inertia_min = 0.4
        self.inertia_max = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_rate = 0.1
        self.temperature = 1.0

    def __call__(self, func):
        self.initialize_particles(func)
        evaluations = 0
        max_velocity = (func.bounds.ub - func.bounds.lb) / 10

        while evaluations < self.budget:
            for i in range(self.population_size):
                score = func(self.positions[i])
                evaluations += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i].copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.positions[i])
                social = self.social_coeff * r2 * (self.global_best_position - self.positions[i])

                # Chaotic inertia weight
                chaos_param = 4 * r1 * (1 - r1)
                self.inertia_weight = self.inertia_min + (self.inertia_max - self.inertia_min) * chaos_param

                self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive + social
                self.velocities[i] = np.clip(self.velocities[i], -max_velocity, max_velocity)
                self.positions[i] += self.velocities[i]

                # Hybrid mutation mechanism
                if np.random.rand() < self.mutation_rate * (1 - evaluations / self.budget):
                    hybrid_mutation = 0.5 * (np.random.rand(self.dim) - 0.5) * (func.bounds.ub - func.bounds.lb) * self.temperature
                    self.positions[i] += hybrid_mutation
                    self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)

                # Dynamic quantum jump based on temperature
                if np.random.rand() < 0.05 * (1 - evaluations / self.budget):
                    quantum_jump = np.random.rand(self.dim) * (func.bounds.ub - func.bounds.lb) * self.temperature
                    self.positions[i] = self.global_best_position + quantum_jump * (np.random.rand(self.dim) - 0.5)
                    self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)

            # Adjust temperature dynamically
            self.temperature = 0.5 + 0.5 * (1 - evaluations / self.budget)

        return self.global_best_position, self.global_best_score

    def initialize_particles(self, func):
        self.positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.array([func(pos) for pos in self.positions])
        self.global_best_position = self.positions[np.argmin(self.personal_best_scores)]
        self.global_best_score = np.min(self.personal_best_scores)