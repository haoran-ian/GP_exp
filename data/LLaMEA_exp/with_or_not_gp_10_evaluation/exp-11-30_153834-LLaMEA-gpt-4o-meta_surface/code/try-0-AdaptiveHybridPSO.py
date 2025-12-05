import numpy as np

class AdaptiveHybridPSO:
    def __init__(self, budget, dim, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = swarm_size
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.pbest = self.positions.copy()
        self.pbest_scores = np.full(self.swarm_size, np.inf)
        self.gbest = self.positions[0].copy()
        self.gbest_score = np.inf
        self.func_evals = 0

    def optimize(self, func):
        inertia_weight = 0.7
        cognitive_weight = 1.5
        social_weight = 1.5
        mutation_factor = 0.5

        while self.func_evals < self.budget:
            for i in range(self.swarm_size):
                if self.func_evals >= self.budget:
                    break

                # Evaluate the current position
                score = func(self.positions[i])
                self.func_evals += 1

                # Update personal best
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest[i] = self.positions[i].copy()

                # Update global best
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest = self.positions[i].copy()

            # Update velocities and positions
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = cognitive_weight * r1 * (self.pbest[i] - self.positions[i])
                social_velocity = social_weight * r2 * (self.gbest - self.positions[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity

                # Update position
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

                # Apply differential mutation if a better global best is found
                if np.random.rand() < mutation_factor:
                    idxs = np.random.choice(self.swarm_size, 3, replace=False)
                    mutated_position = self.positions[idxs[0]] + mutation_factor * (self.positions[idxs[1]] - self.positions[idxs[2]])
                    mutated_position = np.clip(mutated_position, self.lower_bound, self.upper_bound)

                    mutated_score = func(mutated_position)
                    self.func_evals += 1

                    if mutated_score < self.pbest_scores[i]:
                        self.pbest_scores[i] = mutated_score
                        self.pbest[i] = mutated_position.copy()

                    if mutated_score < self.gbest_score:
                        self.gbest_score = mutated_score
                        self.gbest = mutated_position.copy()

        return self.gbest, self.gbest_score

    def __call__(self, func):
        return self.optimize(func)