import numpy as np

class EnhancedHybridPSO_LAM:
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
        self.subpop_size = self.population_size // 2
        self.elite_perturbation_prob = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.positions = lb + (ub - lb) * self.positions
        
        evaluations = 0
        while evaluations < self.budget:
            subpop_indices = np.random.choice(self.population_size, self.subpop_size, replace=False)
            subpop_positions = self.positions[subpop_indices]
            subpop_velocities = self.velocities[subpop_indices]

            for i in range(self.subpop_size):
                value = func(subpop_positions[i])
                evaluations += 1

                if value < self.personal_best_values[subpop_indices[i]]:
                    self.personal_best_values[subpop_indices[i]] = value
                    self.personal_best_positions[subpop_indices[i]] = subpop_positions[i]

                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = subpop_positions[i]

                if evaluations >= self.budget:
                    break

            for i in range(self.subpop_size):
                neighborhood_indices = np.random.choice(self.subpop_size, max(2, self.subpop_size // 5), replace=False)
                neighborhood_best_value = np.inf
                for idx in neighborhood_indices:
                    if self.personal_best_values[subpop_indices[idx]] < neighborhood_best_value:
                        neighborhood_best_value = self.personal_best_values[subpop_indices[idx]]
                        self.neighborhood_best_positions[subpop_indices[i]] = self.personal_best_positions[subpop_indices[idx]]

            eval_ratio = evaluations / self.budget
            self.w = self.w_final + (self.w_init - self.w_final) * (1 - eval_ratio**2)
            self.c1 = self.c1_final + (self.c1_init - self.c1_final) * eval_ratio
            self.c2 = self.c2_final + (self.c2_init - self.c2_final) * (1 - eval_ratio)

            for i in range(self.subpop_size):
                r1, r2 = np.random.rand(2)
                subpop_velocities[i] = (
                    self.w * subpop_velocities[i] + 
                    self.c1 * r1 * (self.personal_best_positions[subpop_indices[i]] - subpop_positions[i]) + 
                    self.c2 * r2 * (self.neighborhood_best_positions[subpop_indices[i]] - subpop_positions[i])
                )
                subpop_positions[i] += subpop_velocities[i]
                subpop_positions[i] = np.clip(subpop_positions[i], lb, ub)

            self.positions[subpop_indices] = subpop_positions
            self.velocities[subpop_indices] = subpop_velocities

            if np.random.rand() < self.mutation_rate:
                idxs = np.random.choice(self.population_size, 3, replace=False)
                target, donor1, donor2 = self.positions[idxs]
                weight = 0.3 + 0.7 * (1 - eval_ratio)
                mutant = target + weight * (donor1 - donor2)
                mutant = np.clip(mutant, lb, ub)
                
                if evaluations < self.budget:
                    mutant_value = func(mutant)
                    evaluations += 1
                    if mutant_value < self.personal_best_values[idxs[0]]:
                        self.personal_best_values[idxs[0]] = mutant_value
                        self.personal_best_positions[idxs[0]] = mutant

            if evaluations < self.budget:
                if np.random.rand() < self.elite_perturbation_prob:
                    elite_indices = np.argsort(self.personal_best_values)[:2]
                    elite_position = self.personal_best_positions[elite_indices[0]]
                    perturbation = np.random.normal(0, 0.01 * (ub - lb), self.dim)
                    perturbed_position = elite_position + perturbation
                    perturbed_position = np.clip(perturbed_position, lb, ub)
                    perturbed_value = func(perturbed_position)
                    evaluations += 1
                    if perturbed_value < self.global_best_value:
                        self.global_best_value = perturbed_value
                        self.global_best_position = perturbed_position

        return self.global_best_position, self.global_best_value