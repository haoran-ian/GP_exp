import numpy as np

class EFTDQSAAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.initial_quantum_coeff = 0.1
        self.quantum_decay = 0.98
        self.learning_rate_decay = 0.98
        self.convergence_threshold = 0.001

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        particles = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(x) for x in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)
        
        evaluations = population_size
        quantum_coeff = self.initial_quantum_coeff
        elite_ratio = 0.2
        elite_count = max(1, int(population_size * elite_ratio))
        
        previous_global_best_score = global_best_score
        
        while evaluations < self.budget:
            for i in range(population_size):
                quantum_exploration = np.random.uniform(-1, 1, self.dim) * quantum_coeff
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                inertia_dyn = self.inertia_weight * (0.5 + 0.5 * (self.budget - evaluations) / self.budget)  # Linear inertia adjustment
                velocities[i] = (inertia_dyn * velocities[i] +
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
            distances = np.linalg.norm(personal_best_positions - elite_average_position, axis=1)
            farthest_indices = np.argsort(-distances)[:elite_count]

            for fi in farthest_indices:
                exploration_vector = np.random.uniform(-1, 1, self.dim)
                particles[fi] = elite_average_position + exploration_vector * quantum_coeff * 0.5  # Elite-guided mutation
                particles[fi] = np.clip(particles[fi], lb, ub)
                score = func(particles[fi])
                evaluations += 1

                if score < personal_best_scores[fi]:
                    personal_best_scores[fi] = score
                    personal_best_positions[fi] = particles[fi]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[fi]

            if evaluations % (population_size * 10) == 0:
                quantum_coeff *= self.quantum_decay
                # Adjust learning rate based on convergence speed
                if abs(previous_global_best_score - global_best_score) < self.convergence_threshold:
                    self.cognitive_coeff *= self.learning_rate_decay
                    self.social_coeff *= self.learning_rate_decay
                previous_global_best_score = global_best_score

            # Dynamically adjust population size
            if evaluations % (population_size * 5) == 0 and evaluations < self.budget * 0.9:
                convergence_rate = np.abs(previous_global_best_score - global_best_score) / previous_global_best_score
                if convergence_rate < self.convergence_threshold:
                    population_size = max(10, int(population_size * 0.9))
                    elite_count = max(1, int(population_size * elite_ratio))
                    particles = particles[:population_size]
                    velocities = velocities[:population_size]
                    personal_best_positions = personal_best_positions[:population_size]
                    personal_best_scores = personal_best_scores[:population_size]

        return global_best_position, global_best_score