import numpy as np

class EnhancedAdaptiveQSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(40, budget // 5)
        self.sub_population_sizes = [self.population_size // 2, self.population_size - self.population_size // 2]
        self.positions = [None, None]
        self.velocities = [None, None]
        self.personal_best_positions = [None, None]
        self.personal_best_scores = [None, None]
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.inertia_weight = 0.9
        self.inertia_min = 0.4
        self.inertia_max = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_rate = 0.1
        self.temperature = 1.0
        self.crossover_rate = 0.5

    def __call__(self, func):
        self.initialize_particles(func)
        evaluations = 0
        max_velocity = (func.bounds.ub - func.bounds.lb) / 10

        while evaluations < self.budget:
            for pop in range(2):
                for i in range(self.sub_population_sizes[pop]):
                    score = func(self.positions[pop][i])
                    evaluations += 1

                    if score < self.personal_best_scores[pop][i]:
                        self.personal_best_scores[pop][i] = score
                        self.personal_best_positions[pop][i] = self.positions[pop][i].copy()

                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.positions[pop][i].copy()

            for pop in range(2):
                for i in range(self.sub_population_sizes[pop]):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    cognitive = self.cognitive_coeff * r1 * (self.personal_best_positions[pop][i] - self.positions[pop][i])
                    social = self.social_coeff * r2 * (self.global_best_position - self.positions[pop][i])

                    learning_rate = (1 - evaluations / self.budget)
                    self.velocities[pop][i] = (self.inertia_weight * self.velocities[pop][i] + cognitive + social) * learning_rate
                    self.velocities[pop][i] = np.clip(self.velocities[pop][i], -max_velocity, max_velocity)
                    self.positions[pop][i] += self.velocities[pop][i]

                    if np.random.rand() < self.mutation_rate * (1 - evaluations / self.budget):
                        mutation = (np.random.rand(self.dim) - 0.5) * (func.bounds.ub - func.bounds.lb) * 0.1 * self.temperature
                        self.positions[pop][i] += mutation
                        self.positions[pop][i] = np.clip(self.positions[pop][i], func.bounds.lb, func.bounds.ub)

                    if np.random.rand() < self.crossover_rate * (1 - evaluations / self.budget):
                        partner_idx = np.random.randint(self.sub_population_sizes[pop])
                        crossover_mask = np.random.rand(self.dim) < 0.5
                        self.positions[pop][i][crossover_mask] = self.positions[pop][partner_idx][crossover_mask]

                    if np.random.rand() < 0.05 * (1 - evaluations / self.budget):
                        quantum_jump = np.random.rand(self.dim) * (func.bounds.ub - func.bounds.lb) * self.temperature
                        self.positions[pop][i] = self.global_best_position + quantum_jump * (np.random.rand(self.dim) - 0.5)
                        self.positions[pop][i] = np.clip(self.positions[pop][i], func.bounds.lb, func.bounds.ub)

            self.inertia_weight = self.inertia_max - ((self.inertia_max - self.inertia_min) * (evaluations / self.budget))**2 + np.random.uniform(-0.05, 0.05)
            self.temperature = 0.5 + 0.5 * (1 - evaluations / self.budget)

        return self.global_best_position, self.global_best_score

    def initialize_particles(self, func):
        for pop in range(2):
            self.positions[pop] = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.sub_population_sizes[pop], self.dim))
            self.velocities[pop] = np.zeros((self.sub_population_sizes[pop], self.dim))
            self.personal_best_positions[pop] = self.positions[pop].copy()
            self.personal_best_scores[pop] = np.array([func(pos) for pos in self.positions[pop]])
        
        combined_scores = np.concatenate(self.personal_best_scores)
        combined_positions = np.concatenate(self.positions)
        self.global_best_position = combined_positions[np.argmin(combined_scores)]
        self.global_best_score = np.min(combined_scores)