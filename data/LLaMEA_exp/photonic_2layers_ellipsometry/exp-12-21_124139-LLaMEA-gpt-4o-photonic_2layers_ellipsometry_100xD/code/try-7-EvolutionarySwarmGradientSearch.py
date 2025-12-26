import numpy as np

class EvolutionarySwarmGradientSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = 50
        self.inertia_weight = 0.9  # Updated: Start with higher inertia weight
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5  # Social coefficient: Dynamic adjustment
        self.mutation_rate = 0.1
        self.inertia_weight_decay = (0.9 - 0.4) / self.budget  # Updated: Adaptive inertia

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize swarm
        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        while self.evaluations < self.budget:
            # Elitism: Save the best position from the previous generation
            previous_global_best_position = np.copy(global_best_position)  # New line for elitism

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Update velocity
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                    + self.social_coeff * np.random.rand(self.dim) * (global_best_position - positions[i])
                )

                # Update position
                positions[i] += velocities[i]

                # Apply mutation
                diversity = np.std(positions)
                self.mutation_rate = max(0.01, min(0.3, diversity * 0.1))  # Adaptive mutation rate
                if np.random.rand() < self.mutation_rate:
                    mutation = np.random.normal(0, 0.1, self.dim)
                    positions[i] += mutation

                # Clip positions to the bounds
                positions[i] = np.clip(positions[i], lb, ub)

                # Evaluate and update personal and global bests
                score = func(positions[i])
                self.evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = score
                    if score < global_best_score:
                        global_best_position = positions[i]
                        global_best_score = score
            
            # Apply adaptive inertia weight
            self.inertia_weight = max(0.4, self.inertia_weight - self.inertia_weight_decay)  # Updated inertia weight
            # Ensure elitism
            if func(previous_global_best_position) < global_best_score:
                global_best_position = previous_global_best_position

            # Dynamically adjust social coefficient based on convergence
            self.social_coeff = min(2.0, self.social_coeff + (global_best_score / (np.mean(personal_best_scores) + 1e-8)) * 0.1)

        return global_best_position