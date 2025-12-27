import numpy as np

class EnhancedHybridPSO:
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
        self.w = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.mutation_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.positions = lb + (ub - lb) * self.positions
        
        evaluations = 0
        while evaluations < self.budget:
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

            r1, r2 = np.random.rand(2)
            for i in range(self.population_size):
                self.velocities[i] = (
                    self.w * self.velocities[i] + 
                    self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) + 
                    self.c2 * r2 * (self.global_best_position - self.positions[i])
                )
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], lb, ub)

            self.w = 0.4 + 0.5 * (1 - evaluations / self.budget)
            self.c1 = 2.0 * (1 - evaluations / self.budget)
            self.c2 = 2.0 - self.c1

            self.mutation_rate = 0.05 + 0.45 * (evaluations / self.budget)
            if np.random.rand() < self.mutation_rate:
                idxs = np.random.choice(self.population_size, 3, replace=False)
                target, donor1, donor2 = self.positions[idxs]
                weight = 0.8 + 0.2 * (1 - evaluations / self.budget)
                mutant = target + weight * (donor1 - donor2)
                mutant = np.clip(mutant, lb, ub)
                
                if evaluations < self.budget:
                    mutant_value = func(mutant)
                    evaluations += 1
                    if mutant_value < self.personal_best_values[idxs[0]]:
                        self.personal_best_values[idxs[0]] = mutant_value
                        self.personal_best_positions[idxs[0]] = mutant

            if evaluations < self.budget:
                local_radius = 0.05 * (ub - lb) * (1 - evaluations / self.budget)
                for _ in range(5):  # increased local search frequency
                    local_search_position = self.global_best_position + np.random.uniform(-local_radius, local_radius, self.dim)
                    local_search_position = np.clip(local_search_position, lb, ub)
                    local_search_value = func(local_search_position)
                    evaluations += 1
                    if local_search_value < self.global_best_value:
                        self.global_best_value = local_search_value
                        self.global_best_position = local_search_position

        return self.global_best_position, self.global_best_value