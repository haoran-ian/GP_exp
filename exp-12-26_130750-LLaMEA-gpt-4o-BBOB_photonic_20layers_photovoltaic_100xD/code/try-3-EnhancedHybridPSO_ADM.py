import numpy as np

class EnhancedHybridPSO_ADM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = int(np.clip(5 * np.log10(dim), 10, 50))
        self.max_population_size = self.initial_population_size * 2
        self.velocities = np.random.rand(self.initial_population_size, dim) * 0.1
        self.positions = np.random.rand(self.initial_population_size, dim)
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.initial_population_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.w_max, self.w_min = 0.9, 0.4  # inertia weights
        self.c1_start, self.c1_end = 2.5, 0.5  # cognitive coefficients
        self.c2_start, self.c2_end = 0.5, 2.5  # social coefficients
        self.mutation_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.positions = lb + (ub - lb) * self.positions
        
        evaluations = 0
        while evaluations < self.budget:
            population_size = int(self.initial_population_size + (self.max_population_size - self.initial_population_size) * (evaluations / self.budget))
            if len(self.positions) < population_size:
                new_positions = lb + (ub - lb) * np.random.rand(population_size - len(self.positions), self.dim)
                new_velocities = np.random.rand(population_size - len(self.velocities), self.dim) * 0.1
                self.positions = np.vstack((self.positions, new_positions))
                self.velocities = np.vstack((self.velocities, new_velocities))
                self.personal_best_positions = np.vstack((self.personal_best_positions, new_positions))
                self.personal_best_values = np.append(self.personal_best_values, np.full(population_size - len(self.personal_best_values), np.inf))
            
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            c1 = self.c1_start - (self.c1_start - self.c1_end) * (evaluations / self.budget)
            c2 = self.c2_start + (self.c2_end - self.c2_start) * (evaluations / self.budget)
            
            for i in range(population_size):
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
            for i in range(population_size):
                self.velocities[i] = (
                    w * self.velocities[i] + 
                    c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) + 
                    c2 * r2 * (self.global_best_position - self.positions[i])
                )
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], lb, ub)

            if np.random.rand() < self.mutation_rate:
                idxs = np.random.choice(population_size, 3, replace=False)
                target, donor1, donor2 = self.positions[idxs]
                mutant = target + 0.8 * (donor1 - donor2)
                mutant = np.clip(mutant, lb, ub)
                
                if evaluations < self.budget:
                    mutant_value = func(mutant)
                    evaluations += 1
                    if mutant_value < self.personal_best_values[idxs[0]]:
                        self.personal_best_values[idxs[0]] = mutant_value
                        self.personal_best_positions[idxs[0]] = mutant

        return self.global_best_position, self.global_best_value