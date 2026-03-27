import numpy as np

class EnhancedHybridPSODEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) * 0.1
        self.personal_best_positions = self.particles.copy()
        self.global_best_position = self.particles[0].copy()
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_value = np.inf
        self.c1, self.c2 = 1.5, 2.0
        self.w = 0.9  # Start with higher inertia weight for exploration
        self.w_min = 0.4  # Minimum inertia weight for exploitation
        self.de_cross_rate = 0.9  # Enhanced crossover to improve exploitation
        self.de_f = 0.8  # Increased mutation factor for diversity

    def adaptive_inertia_weight(self, evaluations):
        return self.w * ((self.budget - evaluations) / self.budget) + self.w_min * (evaluations / self.budget)
    
    def update_velocity(self, evaluations):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        cognitive_component = self.c1 * r1 * (self.personal_best_positions - self.particles)
        social_component = self.c2 * r2 * (self.global_best_position - self.particles)
        inertia_weight = self.adaptive_inertia_weight(evaluations)
        self.velocities = inertia_weight * self.velocities + cognitive_component + social_component

    def update_position(self, bounds):
        self.particles += self.velocities
        self.particles = np.clip(self.particles, bounds.lb, bounds.ub)

    def enhanced_differential_evolution(self, func, bounds):
        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.particles[np.random.choice(indices, 3, replace=False)]
            mutant_vector = np.clip(a + self.de_f * (b - c), bounds.lb, bounds.ub)
            crossover = np.random.rand(self.dim) < self.de_cross_rate
            trial_vector = np.where(crossover, mutant_vector, self.particles[i])
            trial_value = func(trial_vector)
            if trial_value < self.personal_best_values[i]:
                self.personal_best_positions[i] = trial_vector
                self.personal_best_values[i] = trial_value
                if trial_value < self.global_best_value:
                    self.global_best_position = trial_vector
                    self.global_best_value = trial_value

    def __call__(self, func):
        evaluations = 0
        bounds = func.bounds
        while evaluations < self.budget:
            for i in range(self.population_size):
                fitness = func(self.particles[i])
                evaluations += 1
                if fitness < self.personal_best_values[i]:
                    self.personal_best_positions[i] = self.particles[i].copy()
                    self.personal_best_values[i] = fitness
                    if fitness < self.global_best_value:
                        self.global_best_position = self.particles[i].copy()
                        self.global_best_value = fitness
                if evaluations >= self.budget:
                    return self.global_best_position
            self.update_velocity(evaluations)
            self.update_position(bounds)
            self.enhanced_differential_evolution(func, bounds)
        return self.global_best_position