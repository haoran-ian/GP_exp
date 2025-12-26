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
        self.memory_decay = 0.99
        self.mutation_factor = 0.5
        self.crowd_distance_factor = 0.1  # New parameter for crowd-inspired exploration

    def levy_flight(self, L):
        sigma = (np.math.gamma(1 + self.alpha) * np.sin(np.pi * self.alpha / 2) /
                 (np.math.gamma((1 + self.alpha) / 2) * self.alpha * 
                  2 ** ((self.alpha - 1) / 2))) ** (1 / self.alpha)
        u = np.random.normal(0, sigma, size=L)
        v = np.random.normal(0, 1, size=L)
        step = u / abs(v) ** (1 / self.alpha)
        return step

    def dynamic_levy_scale(self, evaluations):
        scale_factor = (self.budget - evaluations) / self.budget
        return scale_factor * self.levy_flight(self.dim)
        
    def update_inertia_weight(self, evaluations):
        return (self.inertia_weight_initial - self.inertia_weight_final) * \
               ((self.budget - evaluations) / self.budget) + self.inertia_weight_final
    
    def differential_mutation(self, target, best, a, b):
        return target + self.mutation_factor * (best - target) + self.mutation_factor * 1.1 * (a - b)  # Adjusted mutation scaling
    
    def crowd_inspired_exploration(self, particles):
        mean_position = np.mean(particles, axis=0)
        distances = np.linalg.norm(particles - mean_position, axis=1)
        max_distance = np.max(distances)
        if max_distance > 0:
            exploration_vector = self.crowd_distance_factor * (particles - mean_position) / max_distance
            return exploration_vector
        return np.zeros_like(particles)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        global_best_memory = global_best_position

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
                
                # Adaptive velocity clipping
                velocities[i] = np.clip(velocities[i], lb - particles[i], ub - particles[i])  

                particles[i] += velocities[i]

                if success_measure < self.success_threshold:
                    particles[i] += self.dynamic_levy_scale(evaluations)
                
                particles[i] = np.clip(particles[i], lb, ub)
                
                score = func(particles[i])
                evaluations += 1
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                    
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = particles[i]
                        global_best_memory = global_best_position + self.dynamic_levy_scale(evaluations)
                
            # Differential Evolution-inspired mutation for global best diversification
            if evaluations < self.budget:
                random_indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = particles[random_indices]
                mutated_gbest = self.differential_mutation(global_best_position, global_best_memory, a, b)
                mutated_gbest = np.clip(mutated_gbest, lb, ub)
                mutated_score = func(mutated_gbest)
                evaluations += 1
                
                if mutated_score < global_best_score:
                    global_best_score = mutated_score
                    global_best_position = mutated_gbest
                    global_best_memory = mutated_gbest

            # Apply crowd-inspired exploration
            exploration_vector = self.crowd_inspired_exploration(particles)
            particles += exploration_vector
            particles = np.clip(particles, lb, ub)

            global_best_position = self.memory_decay * global_best_memory + (1 - self.memory_decay) * global_best_position

        return global_best_position