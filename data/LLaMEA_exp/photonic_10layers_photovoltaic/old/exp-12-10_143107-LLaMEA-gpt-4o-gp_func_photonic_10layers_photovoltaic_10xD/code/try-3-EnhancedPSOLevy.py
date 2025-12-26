import numpy as np

class EnhancedPSOLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.alpha = 1.5  # exponent for Lévy flight distribution
        self.min_learning_rate = 0.1
        self.max_learning_rate = 2.0

    def levy_flight(self, L):
        sigma = (np.math.gamma(1 + self.alpha) * np.sin(np.pi * self.alpha / 2) /
                 (np.math.gamma((1 + self.alpha) / 2) * self.alpha * 
                  2 ** ((self.alpha - 1) / 2))) ** (1 / self.alpha)
        u = np.random.normal(0, sigma, size=L)
        v = np.random.normal(0, 1, size=L)
        step = u / abs(v) ** (1 / self.alpha)
        return step

    def dynamic_learning_rate(self, iteration, max_iterations):
        return self.min_learning_rate + (self.max_learning_rate - self.min_learning_rate) * (1 - (iteration / max_iterations))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size
        iteration = 0
        max_iterations = self.budget // self.population_size

        while evaluations < self.budget:
            diversity = np.mean(np.std(particles, axis=0))
            self.inertia_weight = 0.5 + (0.5 * (diversity / (ub - lb).mean()))
            
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_coefficient = self.cognitive_coefficient * self.dynamic_learning_rate(iteration, max_iterations)
                social_coefficient = self.social_coefficient * self.dynamic_learning_rate(iteration, max_iterations)
                
                cognitive_term = cognitive_coefficient * r1 * (personal_best_positions[i] - particles[i])
                social_term = social_coefficient * r2 * (global_best_position - particles[i])
                velocities[i] = (self.inertia_weight * velocities[i] + cognitive_term + social_term)
                
                particles[i] += velocities[i]
                
                if np.random.rand() < 0.3:
                    particles[i] += self.levy_flight(self.dim)
                
                particles[i] = np.clip(particles[i], lb, ub)
                
                score = func(particles[i])
                evaluations += 1
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                    
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = particles[i]

            iteration += 1

        return global_best_position