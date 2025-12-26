import numpy as np

class EnhancedQuantumLevySwarmOptimizerV8:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.positions = None
        self.velocities = None
        self.best_positions = None
        self.best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.velocity_clamp_factor = 0.1
        self.elite_fraction = 0.1
        self.chaos_factor = 0.1
        self.energy_threshold = 0.8

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta *
                  2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(size) * sigma
        v = np.random.randn(size)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def initialize(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        center = (lb + ub) / 2
        quantum_range = (ub - lb) / 2
        self.positions = np.random.uniform(center - quantum_range, center + quantum_range, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-quantum_range, quantum_range, (self.population_size, self.dim))
        self.best_positions = np.copy(self.positions)
        self.best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = np.random.uniform(lb, ub, self.dim)

    def update_positions_and_velocities(self, bounds, progress_factor):
        lb, ub = bounds.lb, bounds.ub
        inertia_weight = (self.inertia_weight_initial - self.inertia_weight_final) * (1 - progress_factor) + self.inertia_weight_final
        inertia_weight *= (1 + self.chaos_factor * np.sin(np.pi * progress_factor))  # Chaos-induced inertia adjustment

        cognitive_const, social_const = self.dynamic_phase_switch(progress_factor)

        diversity = np.std(self.positions, axis=0).mean()
        mutation_strength = 0.1 * (1 - progress_factor) * diversity
        energy_adjustment = self.energy_based_mutation(progress_factor)

        elite_size = int(self.elite_fraction * self.population_size * (1.0 - progress_factor))
        if progress_factor > 0.9:
            elite_size = int(self.elite_fraction * self.population_size * (1.0 + 0.5 * progress_factor))
        elite_indices = np.argsort(self.best_scores)[:elite_size]
        
        for i in range(self.population_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            cognitive_velocity = cognitive_const * r1 * (self.best_positions[i] - self.positions[i])
            social_velocity = social_const * r2 * (self.global_best_position - self.positions[i])
            levy_vel = self.levy_flight(self.dim)
            quantum_weight = 0.2 * (1 - progress_factor) * (1 + 0.8 * np.sin(np.pi * progress_factor))
            if i in elite_indices:
                inertia_component = inertia_weight * self.velocities[i]
            else:
                inertia_component = 0.5 * self.velocities[i]
            self.velocities[i] = (inertia_component +
                                  cognitive_velocity + social_velocity +
                                  quantum_weight * levy_vel +
                                  mutation_strength * np.random.randn(self.dim) +
                                  energy_adjustment * np.random.randn(self.dim))
            max_velocity = (0.15 - 0.05 * progress_factor) * (ub - lb)
            self.velocities[i] = np.clip(self.velocities[i], -max_velocity, max_velocity)
            self.positions[i] = np.clip(self.positions[i] + self.velocities[i], lb, ub)
            if np.random.rand() < 0.01 + 0.005 * progress_factor:  # Enhanced diversity induction
                self.positions[i] = np.random.uniform(lb, ub, self.dim)

    def dynamic_phase_switch(self, progress_factor):
        if progress_factor < 0.3:
            return 1.5, 1.3
        elif progress_factor < 0.7:
            return 1.2, 1.8
        else:
            return 1.0, 2.0

    def energy_based_mutation(self, progress_factor):
        if progress_factor < self.energy_threshold:
            return 0.1 * (1 - progress_factor)
        else:
            return 0.05 * (progress_factor - self.energy_threshold)

    def evaluate_and_update_best(self, func):
        for i in range(self.population_size):
            score = func(self.positions[i])
            if score < self.best_scores[i]:
                self.best_scores[i] = score
                self.best_positions[i] = self.positions[i].copy()
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.positions[i].copy()

    def __call__(self, func):
        bounds = func.bounds
        self.initialize(bounds)
        evaluations = 0

        while evaluations < self.budget:
            progress_factor = evaluations / self.budget
            self.update_positions_and_velocities(bounds, progress_factor)
            self.evaluate_and_update_best(func)
            evaluations += self.population_size

        return self.global_best_position