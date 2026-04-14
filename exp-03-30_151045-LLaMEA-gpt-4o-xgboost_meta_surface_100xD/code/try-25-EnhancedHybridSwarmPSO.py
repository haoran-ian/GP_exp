import numpy as np

class EnhancedHybridSwarmPSO:
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
        self.F = 0.8
        self.CR = 0.9

    def _adaptive_inertia_weight(self):
        return self.w_max - ((self.w_max - self.w_min) * (self.current_eval / self.budget))

    def _levy_flight(self, position):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v)**(1 / beta)
        return position + 0.01 * step

    def _chaos_perturbation(self, position, bounds):
        beta = 0.3 * (1 - self.current_eval / self.budget)
        z = np.random.standard_cauchy(self.dim)
        chaotic_step = beta * z
        return position + chaotic_step

    def _dynamic_boundary_scaling(self, bounds):
        scaling_factor = 0.1 * (1 - self.current_eval / self.budget)
        return bounds.lb + scaling_factor * (bounds.ub - bounds.lb), bounds.ub - scaling_factor * (bounds.ub - bounds.lb)

    def _differential_evolution_mutation(self, population, index):
        idxs = [idx for idx in range(self.population_size) if idx != index]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        mutant = a + self.F * (b - c)
        return mutant

    def _adaptive_chaotic_map(self, position):
        r = 3.99  # logistic map constant
        chaotic_map = (r * position * (1 - position)) % 1
        return chaotic_map

    def __call__(self, func):
        bounds = func.bounds
        while self.current_eval < self.budget:
            lower_bound, upper_bound = self._dynamic_boundary_scaling(bounds)

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
                mutant = self._differential_evolution_mutation(self.particles, i)
                mutant = self._adaptive_chaotic_map(mutant)
                crossover = np.random.rand(self.dim) < self.CR
                new_particle = np.where(crossover, mutant, self.particles[i])
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      self.c1 * np.random.rand() * (self.personal_best_positions[i] - new_particle) +
                                      self.c2 * np.random.rand() * (self.global_best_position - new_particle))
                new_particle += self.velocities[i]
                new_particle = self._chaos_perturbation(new_particle, bounds)
                new_particle = self._levy_flight(new_particle)
                self.particles[i] = np.clip(new_particle, lower_bound, upper_bound)

        return self.global_best_position, self.global_best_value