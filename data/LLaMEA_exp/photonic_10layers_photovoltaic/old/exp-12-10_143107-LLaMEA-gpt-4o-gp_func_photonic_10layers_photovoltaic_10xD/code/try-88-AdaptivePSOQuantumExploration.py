import numpy as np

class AdaptivePSOQuantumExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.alpha = 1.5
        self.mutation_factor = 0.5
        self.quantum_factor = 0.1  # New parameter for quantum-inspired exploration

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

    def differential_mutation(self, target, best, a, b):
        return target + self.mutation_factor * (best - target) + self.mutation_factor * (a - b)

    def quantum_exploration(self, particles, global_best):
        mean_position = np.mean(particles, axis=0)
        exploration_vector = self.quantum_factor * (global_best - mean_position)
        return exploration_vector

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

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_term = self.cognitive_coefficient * r1 * (personal_best_positions[i] - particles[i])
                social_term = self.social_coefficient * r2 * (global_best_position - particles[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_term + social_term
                
                velocities[i] = np.clip(velocities[i], lb - particles[i], ub - particles[i])
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

            if evaluations < self.budget:
                random_indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = particles[random_indices]
                mutated_gbest = self.differential_mutation(global_best_position, global_best_position, a, b)
                mutated_gbest = np.clip(mutated_gbest, lb, ub)
                mutated_score = func(mutated_gbest)
                evaluations += 1
                
                if mutated_score < global_best_score:
                    global_best_score = mutated_score
                    global_best_position = mutated_gbest

            exploration_vector = self.quantum_exploration(particles, global_best_position)
            particles += exploration_vector
            particles = np.clip(particles, lb, ub)

        return global_best_position