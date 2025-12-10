import numpy as np

class EnhancedPSOLevyDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_coefficient_initial = 2.0
        self.social_coefficient_initial = 2.0
        self.alpha = 1.5  # exponent for Lévy flight distribution
        self.success_threshold = 0.1  # threshold for successful particle movement
        self.f = 0.5  # scaling factor for DE
        self.cr = 0.9  # crossover probability for DE

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

    def differential_evolution(self, particles, i, lb, ub):
        indices = [idx for idx in range(self.population_size) if idx != i]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant_vector = particles[a] + self.f * (particles[b] - particles[c])
        mutant_vector = np.clip(mutant_vector, lb, ub)
        trial_vector = np.copy(particles[i])
        crossover = np.random.rand(self.dim) < self.cr
        trial_vector[crossover] = mutant_vector[crossover]
        return trial_vector

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
                velocities[i] = (inertia_weight * velocities[i] + cognitive_term + social_term)
                
                particles[i] += velocities[i]

                if success_measure < self.success_threshold:
                    particles[i] += self.levy_flight(self.dim)
                
                trial_vector = self.differential_evolution(particles, i, lb, ub)
                new_score = func(trial_vector)
                evaluations += 1
                
                if new_score < personal_best_scores[i]:
                    personal_best_scores[i] = new_score
                    personal_best_positions[i] = trial_vector
                    
                    if new_score < global_best_score:
                        global_best_score = new_score
                        global_best_position = trial_vector

        return global_best_position