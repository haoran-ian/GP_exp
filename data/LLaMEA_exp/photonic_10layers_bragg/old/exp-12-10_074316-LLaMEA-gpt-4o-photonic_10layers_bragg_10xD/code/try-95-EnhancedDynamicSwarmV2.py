import numpy as np

class EnhancedDynamicSwarmV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_swarm_size = 30
        self.positions = np.random.uniform(0, 1, (self.initial_swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.initial_swarm_size, self.dim))
        self.best_positions = self.positions.copy()
        self.best_scores = np.full(self.initial_swarm_size, np.inf)
        self.global_best_position = np.random.uniform(0, 1, self.dim)
        self.global_best_score = np.inf
        self.evaluations = 0
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_weight = 0.7
        self.memory = []

    def levy_flight(self, lam=1.5):
        u = np.random.normal(0, 1, self.dim) * (np.sin(np.pi * lam / 2) / np.pi)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / lam))
        return step

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        neighborhood_size = 5

        while self.evaluations < self.budget:
            for i in range(self.positions.shape[0]):
                if self.evaluations >= self.budget:
                    break

                # Adaptive inertia weight
                self.inertia_weight = 0.5 + 0.4 * (self.evaluations / self.budget)
                
                # Update velocity using local best in neighborhood
                neighbors = np.random.choice(self.positions.shape[0], neighborhood_size, replace=False)
                local_best_position = min(neighbors, key=lambda n: self.best_scores[n])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      self.c1 * np.random.rand(self.dim) * (self.best_positions[i] - self.positions[i]) +
                                      self.c2 * np.random.rand(self.dim) * (self.best_positions[local_best_position] - self.positions[i]))

                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], lb, ub)

                if np.random.rand() < 0.1:
                    step = self.levy_flight()
                    local_pos = self.positions[i] + step
                    local_pos = np.clip(local_pos, lb, ub)
                    local_score = func(local_pos)
                    self.evaluations += 1
                    if local_score < self.best_scores[i]:
                        self.best_scores[i] = local_score
                        self.best_positions[i] = local_pos.copy()
                    if local_score < self.global_best_score:
                        self.global_best_score = local_score
                        self.global_best_position = local_pos.copy()

                score = func(self.positions[i])
                self.evaluations += 1

                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.positions[i].copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()

                # Use historical best memory
                if len(self.memory) > 0 and np.random.rand() < 0.1:
                    memory_choice = np.random.choice(len(self.memory))
                    self.positions[i] = self.memory[memory_choice].copy()
                    self.positions[i] = np.clip(self.positions[i], lb, ub)

            self.memory.append(self.global_best_position.copy())
            if len(self.memory) > 20:
                self.memory.pop(0)

        return self.global_best_position