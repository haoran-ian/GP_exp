import numpy as np

class EnhancedAdaptiveHybridPSOLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30
        self.min_population_size = 10
        self.max_population_size = 50
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.alpha = 1.5  # exponent for Lévy flight distribution
        self.success_threshold = 0.1  # threshold for successful particle movement

    def levy_flight(self, L):
        sigma = (np.math.gamma(1 + self.alpha) * np.sin(np.pi * self.alpha / 2) /
                 (np.math.gamma((1 + self.alpha) / 2) * self.alpha * 
                  2 ** ((self.alpha - 1) / 2))) ** (1 / self.alpha)
        u = np.random.normal(0, sigma, size=L)
        v = np.random.normal(0, 1, size=L)
        step = u / abs(v) ** (1 / self.alpha)
        return step
        
    def update_inertia_weight(self, evaluations):
        return (self.inertia_weight_initial - self.inertia_weight_final) * \
               ((self.budget - evaluations) / self.budget) + self.inertia_weight_final

    def adjust_population(self, evaluations):
        scaling_factor = (1 - evaluations / self.budget)
        return int(self.min_population_size + scaling_factor * (self.max_population_size - self.min_population_size))
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        particles = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = population_size
        while evaluations < self.budget:
            inertia_weight = self.update_inertia_weight(evaluations)
            population_size = self.adjust_population(evaluations)
            if particles.shape[0] != population_size:
                particles = np.resize(particles, (population_size, self.dim))
                velocities = np.resize(velocities, (population_size, self.dim))
                if personal_best_positions.shape[0] > population_size:
                    indices_to_keep = np.argsort(personal_best_scores)[:population_size]
                    personal_best_positions = personal_best_positions[indices_to_keep]
                    personal_best_scores = personal_best_scores[indices_to_keep]
                else:
                    new_indices = population_size - personal_best_positions.shape[0]
                    new_particles = np.random.uniform(lb, ub, (new_indices, self.dim))
                    new_velocities = np.random.uniform(-1, 1, (new_indices, self.dim))
                    particles[-new_indices:] = new_particles
                    velocities[-new_indices:] = new_velocities
                    new_scores = np.array([func(p) for p in new_particles])
                    evaluations += new_indices
                    personal_best_positions = np.vstack((personal_best_positions, new_particles))
                    personal_best_scores = np.append(personal_best_scores, new_scores)
            
            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                
                score_improvement = personal_best_scores[i] - global_best_score
                success_measure = score_improvement / (abs(global_best_score) + np.finfo(float).eps)
                
                cognitive_coefficient = 2.0 * (1 - success_measure)
                social_coefficient = 2.0 * (success_measure + 0.5)

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_term = cognitive_coefficient * r1 * (personal_best_positions[i] - particles[i])
                social_term = social_coefficient * r2 * (global_best_position - particles[i])
                velocities[i] = (inertia_weight * velocities[i] + cognitive_term + social_term)
                
                particles[i] += velocities[i]

                if success_measure < self.success_threshold:
                    particles[i] += self.levy_flight(self.dim)
                
                particles[i] = np.clip(particles[i], lb, ub)
                
                score = func(particles[i])
                evaluations += 1
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                    
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = particles[i] + self.levy_flight(self.dim) # Enhanced global consideration

        return global_best_position