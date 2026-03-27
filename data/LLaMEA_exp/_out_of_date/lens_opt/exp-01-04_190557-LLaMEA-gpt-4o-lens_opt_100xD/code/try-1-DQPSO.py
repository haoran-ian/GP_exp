import numpy as np

class DQPSO:
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
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.random_jump_prob = 0.1

    def __call__(self, func):
        self.initialize_particles(func)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Evaluate the current position
                score = func(self.positions[i])
                evaluations += 1

                # Update personal best
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i].copy()

                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()

            # Adjust inertia, cognitive, and social coefficients dynamically
            self.inertia_weight = 0.4 + 0.3 * np.random.rand()
            self.cognitive_coeff = 1.5 + 0.5 * np.random.rand()
            self.social_coeff = 1.5 + 0.5 * np.random.rand()

            # Update particle velocities and positions
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive = self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.positions[i])
                social = self.social_coeff * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive + social
                self.positions[i] += self.velocities[i]

                # Apply dynamic quantum effect with random exploration
                if np.random.rand() < self.random_jump_prob:  # Dynamic quantum jump
                    global_exploration = np.random.rand(self.dim) * (func.bounds.ub - func.bounds.lb) + func.bounds.lb
                    self.positions[i] = self.global_best_position + global_exploration * (np.random.rand(self.dim) - 0.5)

                # Ensure positions are within bounds
                self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)

        return self.global_best_position, self.global_best_score

    def initialize_particles(self, func):
        self.positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.array([func(pos) for pos in self.positions])
        self.global_best_position = self.positions[np.argmin(self.personal_best_scores)]
        self.global_best_score = np.min(self.personal_best_scores)