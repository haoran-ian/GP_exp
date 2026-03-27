import numpy as np

class EnhancedHybridPSO_AVM:
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
        self.w = 0.7  # increased inertia weight for more exploration
        self.c1 = 1.2  # reduced cognitive coefficient
        self.c2 = 1.7  # increased social coefficient for better convergence
        self.initial_mutation_rate = 0.2
        self.final_mutation_rate = 0.05
        self.mutation_rate = self.initial_mutation_rate

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
                # Adaptive inertia weight to focus on exploration initially and exploitation later
                self.w = 0.9 - (0.5 * (evaluations / self.budget))
                self.velocities[i] = (
                    self.w * self.velocities[i] + 
                    self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) + 
                    self.c2 * r2 * (self.global_best_position - self.positions[i])
                )
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], lb, ub)

            # Adaptive mutation rate, decreasing over time to stabilize convergence
            self.mutation_rate = (self.initial_mutation_rate - self.final_mutation_rate) * ((self.budget - evaluations) / self.budget) + self.final_mutation_rate

            if np.random.rand() < self.mutation_rate:
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

        return self.global_best_position, self.global_best_value