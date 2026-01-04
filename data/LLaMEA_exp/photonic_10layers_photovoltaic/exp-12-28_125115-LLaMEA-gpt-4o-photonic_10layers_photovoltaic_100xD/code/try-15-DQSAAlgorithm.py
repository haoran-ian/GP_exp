import numpy as np

class DQSAAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.initial_quantum_coeff = 0.1
        self.quantum_decay = 0.98
        self.learning_rate_decay = 0.98

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(x) for x in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)
        
        evaluations = self.population_size
        quantum_coeff = self.initial_quantum_coeff
        elite_ratio = 0.2
        elite_count = max(1, int(self.population_size * elite_ratio))

        while evaluations < self.budget:
            for i in range(self.population_size):
                quantum_exploration = np.random.uniform(-1, 1, self.dim) * quantum_coeff
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                inertia_dyn = self.inertia_weight * (1 - evaluations / self.budget)  # Dynamic inertia adjustment
                velocities[i] = (inertia_dyn * velocities[i] +  # Use dynamic inertia
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
            
            # Elite-guided search strategy
            elite_indices = np.argsort(personal_best_scores)[:elite_count]
            elite_average_position = np.mean(personal_best_positions[elite_indices], axis=0)
            distances = np.linalg.norm(personal_best_positions - elite_average_position, axis=1)
            farthest_indices = np.argsort(-distances)[:elite_count]

            for fi in farthest_indices:
                exploration_vector = np.random.uniform(-1, 1, self.dim)
                particles[fi] = elite_average_position + exploration_vector * quantum_coeff
                particles[fi] = np.clip(particles[fi], lb, ub)
                score = func(particles[fi])
                evaluations += 1

                if score < personal_best_scores[fi]:
                    personal_best_scores[fi] = score
                    personal_best_positions[fi] = particles[fi]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[fi]

            # Update parameters adaptively
            if evaluations % (self.population_size * 10) == 0:  # Adjust quantum exploration periodically
                quantum_coeff *= self.quantum_decay
            self.cognitive_coeff *= self.learning_rate_decay
            self.social_coeff *= self.learning_rate_decay

            # Introduce adaptive elite_ratio
            if evaluations % (self.population_size * 20) == 0:
                elite_ratio = 0.1 + (0.3 * evaluations / self.budget)
                elite_count = max(1, int(self.population_size * elite_ratio))

        return global_best_position, global_best_score