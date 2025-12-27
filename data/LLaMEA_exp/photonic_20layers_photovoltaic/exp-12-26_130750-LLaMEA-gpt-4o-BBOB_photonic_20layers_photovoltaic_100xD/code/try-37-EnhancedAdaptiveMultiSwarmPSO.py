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
        self.max_velocity = 0.2 * (func.bounds.ub - func.bounds.lb)

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
            self.w = self.w_final + (self.w_init - self.w_final) * (1 - eval_ratio**2)
            self.c1 = self.c1_final + (self.c1_init - self.c1_final) * eval_ratio
            self.c2 = self.c2_final + (self.c2_init - self.c2_final) * (1 - eval_ratio)

            # Update velocities and positions with clamping
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                neighbor_idx = i // self.swarm_size * self.swarm_size
                neighborhood_best_pos = self.personal_best_positions[neighbor_idx:neighbor_idx + self.swarm_size].min(axis=0)
                self.velocities[i] = np.clip(
                    self.w * self.velocities[i] + 
                    self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) + 
                    self.c2 * r2 * (neighborhood_best_pos - self.positions[i]),
                    -self.max_velocity, self.max_velocity
                )
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], lb, ub)

            # Dynamic neighborhood with differential mutation
            if np.random.rand() < 0.3:
                idxs = np.random.choice(self.population_size, 3, replace=False)
                target, donor1, donor2 = self.positions[idxs]
                mutant = target + 0.8 * (donor1 - donor2)
                mutant = np.clip(mutant, lb, ub)
                
                if evaluations < self.budget:
                    mutant_value = func(mutant)
                    evaluations += 1
                    if mutant_value < self.personal_best_values[idxs[0]]:
                        self.personal_best_values[idxs[0]] = mutant_value
                        self.personal_best_positions[idxs[0]] = mutant

            # Gradient-based local enhancement
            if evaluations < self.budget:
                grad_step_size = 0.01 * (ub - lb) * (1 - eval_ratio)
                for _ in range(3):
                    gradient_estimate = np.random.normal(size=self.dim)
                    local_search_position = self.global_best_position - grad_step_size * gradient_estimate
                    local_search_position = np.clip(local_search_position, lb, ub)
                    local_search_value = func(local_search_position)
                    evaluations += 1
                    if local_search_value < self.global_best_value:
                        self.global_best_value = local_search_value
                        self.global_best_position = local_search_position

        return self.global_best_position, self.global_best_value