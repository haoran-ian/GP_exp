import numpy as np

class AQSEO_EDS_V2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.initial_inertia_weight = 0.9
        self.final_inertia_weight = 0.4
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.quantum_coeff = 0.1
        self.quantum_decay = 0.98
        self.learning_rate_decay = 0.98
        self.elite_ratio = 0.2
        self.mutation_factor = 0.1

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
            inertia_weight = (self.initial_inertia_weight - self.final_inertia_weight) * \
                             ((self.budget - evaluations) / self.budget) + self.final_inertia_weight
            for i in range(self.population_size):
                quantum_exploration = np.random.uniform(-1, 1, self.dim) * self.quantum_coeff
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.social_coeff * r2 * (global_best_position - particles[i]) +
                                 quantum_exploration)
                particles[i] += velocities[i]
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
            for ei in elite_indices:
                if np.random.rand() < self.mutation_factor:
                    mutation_vector = np.random.normal(0, 1, self.dim) * self.mutation_factor
                    particles[ei] = elite_average_position + mutation_vector
                    particles[ei] = np.clip(particles[ei], lb, ub)
                    score = func(particles[ei])
                    evaluations += 1

                    if score < personal_best_scores[ei]:
                        personal_best_scores[ei] = score
                        personal_best_positions[ei] = particles[ei]
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = particles[ei]

            if evaluations % (self.population_size * 10) == 0:
                self.quantum_coeff *= self.quantum_decay
            self.cognitive_coeff *= self.learning_rate_decay
            self.social_coeff *= self.learning_rate_decay

        return global_best_position, global_best_score