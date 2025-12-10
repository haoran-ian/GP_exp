import numpy as np

class EnhancedAdaptiveHybridPSOLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_coefficient_initial = 2.0
        self.social_coefficient_initial = 2.0
        self.alpha = 1.5
        self.success_threshold = 0.1
        self.dynamic_population_toggle = True  # Enable dynamic population adjustment
        self.memory_factor = 0.7  # Factor for memory-based velocity update

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

    def adjust_population_size(self, evaluations):
        return max(5, int(self.population_size * (self.budget - evaluations) / self.budget))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size
        while evaluations < self.budget:
            inertia_weight = self.update_inertia_weight(evaluations)
            
            if self.dynamic_population_toggle:
                self.population_size = self.adjust_population_size(evaluations)
                
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                score_improvement = personal_best_scores[i] - global_best_score
                success_measure = score_improvement / (abs(global_best_score) + np.finfo(float).eps)
                
                cognitive_coefficient = self.cognitive_coefficient_initial * (1 - success_measure)
                social_coefficient = self.social_coefficient_initial * (success_measure + 0.5)

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_term = cognitive_coefficient * r1 * (personal_best_positions[i] - particles[i])
                social_term = social_coefficient * r2 * (global_best_position - particles[i])
                memory_term = self.memory_factor * velocities[i]
                
                velocities[i] = (inertia_weight * velocities[i] + cognitive_term + social_term + memory_term)
                
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
                        global_best_position = particles[i] + self.levy_flight(self.dim)

        return global_best_position