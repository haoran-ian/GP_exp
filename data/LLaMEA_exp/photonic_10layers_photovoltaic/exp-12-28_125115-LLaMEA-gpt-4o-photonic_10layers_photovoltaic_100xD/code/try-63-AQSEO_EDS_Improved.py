import numpy as np

class AQSEO_EDS_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.quantum_coeff = 0.1
        self.quantum_decay = 0.98
        self.learning_rate_decay = 0.98
        self.elite_ratio = 0.2
        self.adaptive_factor = 0.2
        self.mutation_chance = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(x) for x in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluations = self.population_size
        elite_count = max(1, int(self.population_size * self.elite_ratio))

        while evaluations < self.budget:
            for i in range(self.population_size):
                quantum_exploration = np.random.uniform(-1, 1, self.dim) * self.quantum_coeff
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                inertia_dyn = self.inertia_weight * (0.5 + 0.5 * (self.budget - evaluations) / self.budget)
                velocities[i] = (inertia_dyn * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best_positions[i] - particles[i]) +
                                 (self.social_coeff * r2 ** 2) * (global_best_position - particles[i]) +
                                 quantum_exploration)
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

                if np.random.rand() < self.mutation_chance:
                    mutation_vector = np.random.normal(0, 0.1, self.dim)
                    particles[i] += mutation_vector
                    particles[i] = np.clip(particles[i], lb, ub)

                score = func(particles[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]

            elite_indices = np.argsort(personal_best_scores)[:elite_count]
            elite_average_position = np.mean(personal_best_positions[elite_indices], axis=0)
            distances = np.linalg.norm(personal_best_positions - elite_average_position, axis=1)
            farthest_indices = np.argsort(-distances)[:elite_count]

            for fi in farthest_indices:
                exploration_vector = np.random.uniform(-1, 1, self.dim) * self.adaptive_factor
                particles[fi] = elite_average_position + exploration_vector * self.quantum_coeff * 0.5
                particles[fi] = np.clip(particles[fi], lb, ub)
                score = func(particles[fi])
                evaluations += 1

                if score < personal_best_scores[fi]:
                    personal_best_scores[fi] = score
                    personal_best_positions[fi] = particles[fi]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[fi]

            if evaluations % (self.population_size * 10) == 0:
                self.quantum_coeff *= self.quantum_decay
            self.cognitive_coeff *= self.learning_rate_decay
            self.social_coeff *= self.learning_rate_decay
            self.adaptive_factor *= self.learning_rate_decay

        return global_best_position, global_best_score