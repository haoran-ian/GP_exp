import numpy as np

class EnhancedAdaptiveMultiSwarmPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = int(np.clip(5 * np.log10(dim), 10, 50))
        self.swarm_count = 3
        self.swarm_size = self.population_size // self.swarm_count
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.positions = np.random.rand(self.population_size, dim)
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.w_init, self.w_final = 0.9, 0.4
        self.c1_init, self.c1_final = 2.5, 0.5
        self.c2_init, self.c2_final = 0.5, 2.5
        self.diversity_threshold = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.positions = lb + (ub - lb) * self.positions
        evaluations = 0

        while evaluations < self.budget:
            # Evaluate current population
            for i in range(self.population_size):
                value = func(self.positions[i])
                evaluations += 1

                if value < self.personal_best_values[i]:
                    self.personal_best_values[i] = value
                    self.personal_best_positions[i] = self.positions[i]

                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = self.positions[i]

                if evaluations >= self.budget:
                    break

            # Self-adaptive inertia weight and coefficients
            eval_ratio = evaluations / self.budget
            self.w = self.w_final + (self.w_init - self.w_final) * np.exp(-2 * eval_ratio)
            self.c1 = self.c1_final + (self.c1_init - self.c1_final) * eval_ratio
            self.c2 = self.c2_final + (self.c2_init - self.c2_final) * (1 - eval_ratio)

            # Update velocities and positions
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                neighbor_idx = i // self.swarm_size * self.swarm_size
                neighborhood_best_pos = self.personal_best_positions[neighbor_idx:neighbor_idx + self.swarm_size].min(axis=0)
                self.velocities[i] = (
                    self.w * self.velocities[i] + 
                    self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) + 
                    self.c2 * r2 * (neighborhood_best_pos - self.positions[i])
                )
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], lb, ub)

            # Introduce diversity mechanism
            if evaluations < self.budget:
                diversity = np.std(self.positions, axis=0).mean()
                if diversity < self.diversity_threshold:
                    for i in range(self.population_size):
                        if np.random.rand() < 0.1:
                            perturbation = np.random.randn(self.dim) * 0.1 * (ub - lb)
                            self.positions[i] = np.clip(self.positions[i] + perturbation, lb, ub)

            # Improved local search around global best
            if evaluations < self.budget:
                local_radius = 0.05 * (ub - lb) * (1 - eval_ratio)
                for _ in range(3):
                    local_search_position = self.global_best_position + np.random.uniform(-local_radius, local_radius, self.dim)
                    local_search_position = np.clip(local_search_position, lb, ub)
                    local_search_value = func(local_search_position)
                    evaluations += 1
                    if local_search_value < self.global_best_value:
                        self.global_best_value = local_search_value
                        self.global_best_position = local_search_position

        return self.global_best_position, self.global_best_value