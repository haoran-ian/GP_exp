import numpy as np

class QuantumDynamicSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.personal_best = self.particles.copy()
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best = None
        self.global_best_score = float('inf')
        self.rotation_angle = np.pi / 4  # Initial rotation angle for quantum-inspired mechanics

    def quantum_rotation(self, vector):
        rotation_matrix = np.array([[np.cos(self.rotation_angle), -np.sin(self.rotation_angle)],
                                    [np.sin(self.rotation_angle), np.cos(self.rotation_angle)]])
        rotated_vector = np.dot(rotation_matrix, vector)
        return rotated_vector

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                score = func(self.particles[i])

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = self.particles[i].copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.particles[i].copy()

            inertia_weight = 0.9 - (0.8 * eval_count / self.budget) + 0.1 * np.cos(np.pi * eval_count / self.budget)
            cognitive_component = np.random.rand(self.population_size, self.dim)
            social_component = np.random.rand(self.population_size, self.dim)
            self.velocities = (
                inertia_weight * self.velocities
                + cognitive_component * (self.personal_best - self.particles)
                + social_component * (self.global_best - self.particles)
            )
            self.particles += self.velocities
            self.particles = np.clip(self.particles, lower_bound, upper_bound)

            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = self.personal_best[a] + 0.8 * (self.personal_best[b] - self.personal_best[c])
                mutant_vector = np.clip(mutant_vector, lower_bound, upper_bound)

                trial_vector = np.where(
                    np.random.rand(self.dim) < 0.9,
                    mutant_vector,
                    self.particles[i]
                )

                trial_vector = self.quantum_rotation(trial_vector)
                trial_vector = np.clip(trial_vector, lower_bound, upper_bound)

                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best[i] = trial_vector.copy()
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best = trial_vector.copy()

        return self.global_best