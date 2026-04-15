import numpy as np

class AdvancedStochasticTunnelingPSO:
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
        self.c1_initial = 2.0
        self.c2_initial = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.current_eval = 0
        self.F = 0.8
        self.CR = 0.9
        self.tunneling_factor = 0.9  # New parameter for stochastic tunneling

    def _adaptive_inertia_weight(self):
        return self.w_max - ((self.w_max - self.w_min) * (self.current_eval / self.budget))

    def _stochastic_tunneling(self, value):
        """Apply stochastic tunneling to enhance exploration."""
        return np.exp(-self.tunneling_factor * (value - self.global_best_value))
    
    def _adaptive_crossover_rate(self):
        return 0.7 + 0.3 * (self.budget - self.current_eval) / self.budget

    def _dynamic_population_size(self):
        return max(20, int(self.population_size * (1 - self.current_eval / self.budget)))

    def _adjust_learning_rates(self):
        progress = self.current_eval / self.budget
        self.c1 = self.c1_initial * (1 - progress)
        self.c2 = self.c2_initial * progress

    def __call__(self, func):
        bounds = func.bounds
        while self.current_eval < self.budget:
            lower_bound = bounds.lb
            upper_bound = bounds.ub
            self.population_size = self._dynamic_population_size()
            self._adjust_learning_rates()

            for i in range(self.population_size):
                value = func(self.particles[i])
                self.current_eval += 1

                # Apply stochastic tunneling to the evaluated value
                tunneled_value = self._stochastic_tunneling(value)

                if tunneled_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = tunneled_value
                    self.personal_best_positions[i] = self.particles[i]

                if tunneled_value < self.global_best_value:
                    self.global_best_value = tunneled_value
                    self.global_best_position = self.particles[i]

            inertia_weight = self._adaptive_inertia_weight()

            for i in range(self.population_size):
                mutant = self._differential_evolution_mutation(self.particles, i)
                crossover = np.random.rand(self.dim) < self._adaptive_crossover_rate()
                new_particle = np.where(crossover, mutant, self.particles[i])
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      self.c1 * np.random.rand() * (self.personal_best_positions[i] - new_particle) +
                                      self.c2 * np.random.rand() * (self.global_best_position - new_particle))
                new_particle += self.velocities[i]
                self.particles[i] = np.clip(new_particle, lower_bound, upper_bound)

        return self.global_best_position, self.global_best_value