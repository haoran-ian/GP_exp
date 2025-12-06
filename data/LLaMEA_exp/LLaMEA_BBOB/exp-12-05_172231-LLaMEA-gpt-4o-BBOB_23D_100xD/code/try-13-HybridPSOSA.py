import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.global_best_position = np.copy(self.particles[0])
        self.personal_best_scores = np.full(self.pop_size, float('inf'))
        self.global_best_score = float('inf')

    def __call__(self, func):
        eval_count = 0
        w, c1, c2 = 0.5, 1.5, 1.5  # PSO parameters
        T = 1.0  # Initial temperature for SA
        alpha = 0.99  # Cooling rate for SA

        while eval_count < self.budget:
            for i in range(self.pop_size):
                # Evaluate current position
                score = func(self.particles[i])
                eval_count += 1

                # Update personal best
                if score < self.personal_best_scores[i] or np.random.rand() < 0.1:  # 10% chance to update
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = np.copy(self.particles[i])

                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = np.copy(self.particles[i])

            for i in range(self.pop_size):
                # PSO update
                r1, r2 = np.random.rand(2)
                c1_i = c1 * np.random.rand(self.dim)  # Dimension-wise random learning factor
                c2_i = c2 * np.random.rand(self.dim)  # Dimension-wise random learning factor
                w_adapted = 0.5 + 0.5 * (eval_count / self.budget)  # Adaptive inertia weight
                self.velocities[i] = (w_adapted * self.velocities[i] +
                                      c1_i * (self.personal_best_positions[i] - self.particles[i]) +
                                      c2_i * (self.global_best_position - self.particles[i]))
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

                # Simulated Annealing perturbation
                candidate_position = self.particles[i] + np.random.normal(0, T, self.dim)
                candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
                candidate_score = func(candidate_position)
                eval_count += 1

                # Acceptance probability
                if candidate_score < self.personal_best_scores[i] or np.random.rand() < np.exp((self.personal_best_scores[i] - candidate_score) / T):
                    self.personal_best_positions[i] = candidate_position
                    self.personal_best_scores[i] = candidate_score

                    if candidate_score < self.global_best_score:
                        self.global_best_score = candidate_score
                        self.global_best_position = candidate_position

            # Adaptive cooling
            T *= alpha * (1 - (eval_count / self.budget)**2)

        return self.global_best_position, self.global_best_score