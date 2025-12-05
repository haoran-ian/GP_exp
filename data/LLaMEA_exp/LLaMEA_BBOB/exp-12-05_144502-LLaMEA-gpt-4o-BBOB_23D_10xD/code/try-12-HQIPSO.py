import numpy as np

class HQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(5, int(np.sqrt(budget)))  # Dynamic population size
        self.positions = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.pop_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        w = 0.5  # Inertia weight
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        beta = 0.8  # Quantum-inspired parameter (adjusted)

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                
                score = func(self.positions[i])
                self.evaluations += 1

                # Update personal best
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]

                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            # Update velocities and positions
            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                # Quantum-inspired movement
                quantum_move = beta * np.sin(2 * np.pi * r1) * (self.global_best_position - self.positions[i])

                # Standard PSO updates
                cognitive_velocity = c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = c2 * r2 * (self.global_best_position - self.positions[i])
                
                self.velocities[i] = w * self.velocities[i] + cognitive_velocity + social_velocity + quantum_move
                self.positions[i] += 1.2 * self.velocities[i]  # Velocity scaling factor introduced
                
                # Ensure positions are within bounds
                self.positions[i] = np.clip(self.positions[i], -5.0, 5.0)
        
        return self.global_best_position, self.global_best_score