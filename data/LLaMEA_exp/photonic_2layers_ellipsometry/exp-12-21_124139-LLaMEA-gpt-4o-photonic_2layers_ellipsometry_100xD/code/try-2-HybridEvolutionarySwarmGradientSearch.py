import numpy as np

class HybridEvolutionarySwarmGradientSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = 50
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_rate = 0.1
        self.inertia_weight_decay = (0.9 - 0.4) / self.budget
        self.crossover_rate = 0.3  # New parameter for crossover

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
            previous_global_best_position = np.copy(global_best_position)

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
                if np.random.rand() < self.mutation_rate:
                    mutation = np.random.normal(0, 0.1, self.dim)
                    positions[i] += mutation

                # Apply crossover
                if np.random.rand() < self.crossover_rate:
                    partner_idx = np.random.randint(self.population_size)
                    crossover_mask = np.random.rand(self.dim) < 0.5
                    positions[i][crossover_mask] = positions[partner_idx][crossover_mask]

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
            self.inertia_weight = max(0.4, self.inertia_weight - self.inertia_weight_decay)

            # Ensure elitism
            if func(previous_global_best_position) < global_best_score:
                global_best_position = previous_global_best_position

        return global_best_position