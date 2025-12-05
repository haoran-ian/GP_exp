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
        c1 = 1.5  # Cognitive coefficient
        
        while self.evaluations < self.budget:
            beta = 0.8 + 0.2 * (1 - self.evaluations / self.budget)  # Adaptive beta
            w = 0.4 + 0.2 * (1 - self.evaluations / self.budget)  # Adaptive inertia weight
            c2 = 0.5 + 2.0 * (self.evaluations / self.budget)  # Adaptive social coefficient
            damping_factor = 0.95 * (0.9 + 0.1 * (self.evaluations / self.budget))  # Adaptive damping factor

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
                if score < self.global_best_score and np.linalg.norm(self.positions[i] - self.global_best_position) > 0.1:
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
                
                self.velocities[i] *= damping_factor  # Apply damping factor to velocity
                self.velocities[i] = w * self.velocities[i] + cognitive_velocity + social_velocity + quantum_move
                self.positions[i] += self.velocities[i] + 0.001 * np.random.randn(self.dim)  # Mutation in velocity update
                
                # Ensure positions are within bounds
                self.positions[i] = np.clip(self.positions[i], -5.0, 5.0)
        
        return self.global_best_position, self.global_best_score