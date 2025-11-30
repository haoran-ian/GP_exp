import numpy as np

class EDIASR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 10
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.4
        self.social_coeff = 1.4
        self.convergence_factor = 0.9  # Added

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
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_coeff * r2 * (global_best_position - positions[i]))
                
                positions[i] = positions[i] + velocities[i]
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

            self.inertia_weight *= self.convergence_factor  # Modified
            self.cognitive_coeff = 1.2 + 0.2 * (1 - evaluations / self.budget)  # Added
            self.social_coeff = 1.2 + 0.2 * (evaluations / self.budget)  # Added
        
        return global_best_position, global_best_score