import numpy as np

class DynamicSubgroupingAdaptivePSO:
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
        self.mutation_factor = 0.8

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

            eval_ratio = evaluations / self.budget
            self.w = self.w_final + (self.w_init - self.w_final) * np.exp(-2 * eval_ratio)
            self.c1 = self.c1_final + (self.c1_init - self.c1_final) * (1 - eval_ratio)
            self.c2 = self.c2_final + (self.c2_init - self.c2_final) * eval_ratio

            # Dynamic subgrouping
            dynamic_swarm_assignments = np.random.permutation(self.population_size)
            for i in range(self.population_size):
                swarm_id = dynamic_swarm_assignments[i] // self.swarm_size
                r1, r2 = np.random.rand(2)
                subgroup_start = swarm_id * self.swarm_size
                subgroup_positions = self.personal_best_positions[subgroup_start:subgroup_start + self.swarm_size]
                subgroup_best_pos = subgroup_positions[np.argmin(self.personal_best_values[subgroup_start:subgroup_start + self.swarm_size])]
                self.velocities[i] = (
                    self.w * self.velocities[i] + 
                    self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) + 
                    self.c2 * r2 * (subgroup_best_pos - self.positions[i])
                )
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], lb, ub)

            if np.random.rand() < 0.3:
                idxs = np.random.choice(self.population_size, 3, replace=False)
                target, donor1, donor2 = self.positions[idxs]
                mutation_factor_adaptive = 0.5 + 0.3 * np.sin(np.pi * eval_ratio)
                mutant = target + mutation_factor_adaptive * (donor1 - donor2)
                mutant = np.clip(mutant, lb, ub)
                
                if evaluations < self.budget:
                    mutant_value = func(mutant)
                    evaluations += 1
                    if mutant_value < self.personal_best_values[idxs[0]]:
                        self.personal_best_values[idxs[0]] = mutant_value
                        self.personal_best_positions[idxs[0]] = mutant

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