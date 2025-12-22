import numpy as np

class EnhancedAdaptiveHybridPSOLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.alpha = 1.5  # exponent for Lévy flight distribution
        self.success_threshold = 0.1
        
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
    
    def update_population_size(self, evaluations):
        return int(self.initial_population_size * (1 - evaluations / self.budget) + 10)
    
    def calculate_swarm_diversity(self, particles):
        return np.mean(np.std(particles, axis=0))
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.update_population_size(0)
        
        particles = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = population_size
        while evaluations < self.budget:
            population_size = self.update_population_size(evaluations)
            inertia_weight = self.update_inertia_weight(evaluations)
            swarm_diversity = self.calculate_swarm_diversity(particles)
            
            cognitive_coefficient = 2.0 * (1 - swarm_diversity)
            social_coefficient = 2.0 * (swarm_diversity + 0.5)
            
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_term = cognitive_coefficient * r1 * (personal_best_positions[i] - particles[i])
                social_term = social_coefficient * r2 * (global_best_position - particles[i])
                velocities[i] = (inertia_weight * velocities[i] + cognitive_term + social_term)
                
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
                        global_best_position = particles[i] + self.levy_flight(self.dim)

        return global_best_position