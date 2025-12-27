import numpy as np

class DynamicQuantumPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = int(np.clip(5 * np.log10(dim), 10, 50))
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.positions = np.random.rand(self.population_size, dim)
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.neighborhood_best_positions = np.copy(self.positions)
        self.w_init, self.w_final = 0.9, 0.4
        self.c1_init, self.c1_final = 2.5, 0.5
        self.c2_init, self.c2_final = 0.5, 2.5
        self.mutation_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.positions = lb + (ub - lb) * self.positions
        
        evaluations = 0
        neighborhood_size = max(2, self.population_size // 10)
        
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

            # Quantum-behavior inspired by neighborhood best
            for i in range(self.population_size):
                neighborhood_indices = np.random.choice(self.population_size, neighborhood_size, replace=False)
                neighborhood_best_value = np.inf
                for idx in neighborhood_indices:
                    if self.personal_best_values[idx] < neighborhood_best_value:
                        neighborhood_best_value = self.personal_best_values[idx]
                        self.neighborhood_best_positions[i] = self.personal_best_positions[idx]

            # Self-adaptive inertia weight and coefficients
            eval_ratio = evaluations / self.budget
            self.w = self.w_final + (self.w_init - self.w_final) * (1 - eval_ratio**2)
            self.c1 = self.c1_final + (self.c1_init - self.c1_final) * eval_ratio
            self.c2 = self.c2_final + (self.c2_init - self.c2_final) * (1 - eval_ratio)

            # Update velocities and positions
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (
                    self.w * self.velocities[i] +
                    self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) +
                    self.c2 * r2 * (self.neighborhood_best_positions[i] - self.positions[i])
                )
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], lb, ub)

            # Adaptive mutation strategy with DE refinement
            self.mutation_rate = 0.2 + 0.3 * eval_ratio
            weight = 0.5 + 0.3 * (1 - eval_ratio)
            for _ in range(3):
                idxs = np.random.choice(self.population_size, 3, replace=False)
                target, donor1, donor2 = self.positions[idxs]
                mutant = target + weight * (donor1 - donor2)
                mutant = np.clip(mutant, lb, ub)
                
                if evaluations < self.budget:
                    mutant_value = func(mutant)
                    evaluations += 1
                    if mutant_value < self.personal_best_values[idxs[0]]:
                        self.personal_best_values[idxs[0]] = mutant_value
                        self.personal_best_positions[idxs[0]] = mutant

            # Quantum-inspired local search refinement
            if evaluations < self.budget:
                local_radius = 0.05 * (ub - lb) * (1 - eval_ratio)
                for _ in range(5):
                    theta = np.random.uniform(0, 2 * np.pi, self.dim)
                    local_search_position = self.global_best_position + local_radius * np.cos(theta)
                    local_search_position = np.clip(local_search_position, lb, ub)
                    local_search_value = func(local_search_position)
                    evaluations += 1
                    if local_search_value < self.global_best_value:
                        self.global_best_value = local_search_value
                        self.global_best_position = local_search_position

        return self.global_best_position, self.global_best_value