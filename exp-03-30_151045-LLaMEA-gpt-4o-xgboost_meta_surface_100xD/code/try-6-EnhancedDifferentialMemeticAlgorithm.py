import numpy as np

class EnhancedDifferentialMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.randn(self.population_size, dim)
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_value = np.inf
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_max = 0.9
        self.w_min = 0.4
        self.current_eval = 0
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability

    def _adaptive_inertia_weight(self):
        return self.w_max - ((self.w_max - self.w_min) * (self.current_eval / (1.1 * self.budget)))
        
    def _adaptive_local_search(self, particle, func, bounds):
        step_size = 0.1 * (bounds.ub - bounds.lb)
        local_best = particle
        local_best_value = func(local_best)
        trials = 5 + int(5 * (1 - (self.current_eval / self.budget)))
        for _ in range(trials):
            candidate = local_best + np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            candidate_value = func(candidate)
            if candidate_value < local_best_value:
                local_best = candidate
                local_best_value = candidate_value
        return local_best, local_best_value
        
    def __call__(self, func):
        bounds = func.bounds
        while self.current_eval < self.budget:
            for i in range(self.population_size):
                value = func(self.particles[i])
                self.current_eval += 1
                
                if value < self.personal_best_values[i]:
                    self.personal_best_values[i] = value
                    self.personal_best_positions[i] = self.particles[i]
                
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = self.particles[i]
            
            inertia_weight = self._adaptive_inertia_weight()
            
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                # Differential mutation
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.particles[idxs]
                mutant = a + self.F * (b - c)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.particles[i])
                trial = np.clip(trial, bounds.lb, bounds.ub)

                trial_value = func(trial)
                self.current_eval += 1

                if trial_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = trial_value
                    self.personal_best_positions[i] = trial
                if trial_value < self.global_best_value:
                    self.global_best_value = trial_value
                    self.global_best_position = trial

                # Update velocity and position
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i]) +
                                      self.c2 * r2 * (self.global_best_position - self.particles[i]) * np.random.uniform(0.9, 1.1))
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], bounds.lb, bounds.ub)

            for i in np.random.choice(self.population_size, self.population_size // 10, replace=False):
                local_best, local_best_value = self._adaptive_local_search(self.particles[i], func, bounds)
                if local_best_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = local_best_value
                    self.personal_best_positions[i] = local_best
                if local_best_value < self.global_best_value:
                    self.global_best_value = local_best_value
                    self.global_best_position = local_best

        return self.global_best_position, self.global_best_value