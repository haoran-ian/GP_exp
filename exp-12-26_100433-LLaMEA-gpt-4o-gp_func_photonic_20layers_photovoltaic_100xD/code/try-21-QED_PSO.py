import numpy as np

class QED_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.inertia_init = 0.9
        self.inertia_final = 0.4
        self.cognitive_init = 2.0
        self.cognitive_final = 1.0
        self.social_init = 2.0
        self.social_final = 1.0
        self.evaluations = 0
        self.quantum_prob = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.pop_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        for iteration in range(self.budget // self.pop_size):
            progress = iteration / (self.budget / self.pop_size)
            inertia = self.inertia_init * (1 - progress) + self.inertia_final * progress
            cognitive = self.cognitive_init * (1 - progress) + self.cognitive_final * progress
            social = self.social_init * (1 - progress) + self.social_final * progress

            for i in range(self.pop_size):
                if self.evaluations < self.budget:
                    score = func(particles[i])
                    self.evaluations += 1

                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = particles[i]

                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = particles[i]
                else:
                    break

            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = (inertia * velocities
                          + cognitive * r1 * (personal_best_positions - particles)
                          + social * r2 * (global_best_position - particles))

            particles += velocities

            self.quantum_prob = 0.1 * (1 - progress) + 0.3 * progress  # Adaptive quantum probability
            if np.random.rand() < self.quantum_prob:
                center = (personal_best_positions + global_best_position) / 2
                delta = np.abs(global_best_position - particles)
                particles = center + np.random.uniform(-1, 1, (self.pop_size, self.dim)) * delta / np.sqrt(2)

            if np.random.rand() < 0.2:
                perturbation = np.random.normal(0, 0.2, (self.pop_size, self.dim))  # Enhanced perturbation
                particles = np.clip(particles + perturbation, lb, ub)
            else:
                particles = np.clip(particles, lb, ub)

        return global_best_position, global_best_score