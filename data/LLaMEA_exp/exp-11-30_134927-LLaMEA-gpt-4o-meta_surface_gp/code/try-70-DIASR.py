import numpy as np

class DIASR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 10
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.4
        self.social_coeff = 1.4
        self.levy_exponent = 1.5  # Added

    def levy_flight(self, size):  # Added
        return np.random.standard_normal(size) * np.random.pareto(self.levy_exponent, size)

    def __call__(self, func):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)
        
        evaluations = self.swarm_size
        
        while evaluations < self.budget:
            for i in range(self.swarm_size):
                r1, r2 = np.random.random(2)
                self.cognitive_coeff = 0.5 + 0.9 * (self.budget - evaluations) / self.budget  # Modified
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_coeff * r2 * (global_best_position - positions[i]))
                
                positions[i] = positions[i] + velocities[i] + self.levy_flight(self.dim)  # Modified
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)
                
                score = func(positions[i])
                evaluations += 1
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]
                
                if evaluations >= self.budget:
                    break

            self.inertia_weight *= 0.99
        
        return global_best_position, global_best_score