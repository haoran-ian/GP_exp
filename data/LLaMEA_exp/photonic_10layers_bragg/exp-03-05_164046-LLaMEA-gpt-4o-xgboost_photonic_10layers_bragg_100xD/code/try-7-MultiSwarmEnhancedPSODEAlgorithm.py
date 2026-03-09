import numpy as np

class MultiSwarmEnhancedPSODEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.num_swarms = 5  # Number of sub-swarms
        self.particles_per_swarm = self.population_size // self.num_swarms
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) * 0.1
        self.personal_best_positions = self.particles.copy()
        self.global_best_positions = [self.particles[i].copy() for i in range(self.num_swarms)]
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_values = np.full(self.num_swarms, np.inf)
        self.c1, self.c2 = 1.5, 2.0  # Initial cognitive and social coefficients
        self.w = 0.5  # Initial inertia weight
        self.de_cross_rate = 0.8
        self.de_f = 0.5
        self.evaluations = 0
        self.diversity_threshold = 0.1
        self.regrouping_threshold = 0.05

    def update_velocity(self, swarm_id):
        start = swarm_id * self.particles_per_swarm
        end = start + self.particles_per_swarm
        r1 = np.random.rand(self.particles_per_swarm, self.dim)
        r2 = np.random.rand(self.particles_per_swarm, self.dim)
        cognitive_component = self.c1 * r1 * (self.personal_best_positions[start:end] - self.particles[start:end])
        social_component = self.c2 * r2 * (self.global_best_positions[swarm_id] - self.particles[start:end])
        self.velocities[start:end] = self.w * self.velocities[start:end] + cognitive_component + social_component
        self.velocities[start:end] = np.clip(self.velocities[start:end], -0.1, 0.1)

    def update_position(self, bounds, swarm_id):
        start = swarm_id * self.particles_per_swarm
        end = start + self.particles_per_swarm
        self.particles[start:end] += self.velocities[start:end]
        self.particles[start:end] = np.clip(self.particles[start:end], bounds.lb, bounds.ub)

    def measure_diversity(self):
        centroid = np.mean(self.particles, axis=0)
        diversity = np.mean(np.linalg.norm(self.particles - centroid, axis=1))
        return diversity

    def regroup_swarms(self, bounds):
        if self.measure_diversity() < self.regrouping_threshold:
            self.particles = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
            self.personal_best_positions = self.particles.copy()
            self.personal_best_values = np.full(self.population_size, np.inf)
            self.global_best_values = np.full(self.num_swarms, np.inf)

    def differential_evolution(self, func, bounds, swarm_id):
        start = swarm_id * self.particles_per_swarm
        end = start + self.particles_per_swarm
        for i in range(start, end):
            indices = [idx for idx in range(start, end) if idx != i]
            a, b, c = self.particles[np.random.choice(indices, 3, replace=False)]
            de_factor = np.random.uniform(0.4, 0.9)
            mutant_vector = np.clip(a + de_factor * (b - c), bounds.lb, bounds.ub)
            crossover = np.random.rand(self.dim) < self.de_cross_rate
            trial_vector = np.where(crossover, mutant_vector, self.particles[i])
            trial_value = func(trial_vector)
            self.evaluations += 1
            if trial_value < self.personal_best_values[i]:
                self.personal_best_positions[i] = trial_vector
                self.personal_best_values[i] = trial_value
                swarm_index = i // self.particles_per_swarm
                if trial_value < self.global_best_values[swarm_index]:
                    self.global_best_positions[swarm_index] = trial_vector
                    self.global_best_values[swarm_index] = trial_value
            if self.evaluations >= self.budget:
                return

    def local_search(self, func, bounds, swarm_id):
        start = swarm_id * self.particles_per_swarm
        end = start + self.particles_per_swarm
        perturbation = np.random.normal(0, 0.005, size=self.dim)
        for i in range(start, end):
            candidate = self.particles[i] + perturbation
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            candidate_value = func(candidate)
            self.evaluations += 1
            if candidate_value < self.personal_best_values[i]:
                self.personal_best_positions[i] = candidate
                self.personal_best_values[i] = candidate_value
                swarm_index = i // self.particles_per_swarm
                if candidate_value < self.global_best_values[swarm_index]:
                    self.global_best_positions[swarm_index] = candidate
                    self.global_best_values[swarm_index] = candidate_value
            if self.evaluations >= self.budget:
                return

    def enhance_diversity(self, bounds):
        for i in range(self.population_size):
            if np.random.rand() < 0.1:
                self.particles[i] = np.random.uniform(bounds.lb, bounds.ub, self.dim)

    def __call__(self, func):
        bounds = func.bounds
        self.evaluations = 0
        while self.evaluations < self.budget:
            for swarm_id in range(self.num_swarms):
                self.update_velocity(swarm_id)
                self.update_position(bounds, swarm_id)
                self.differential_evolution(func, bounds, swarm_id)
                self.local_search(func, bounds, swarm_id)
            if self.measure_diversity() < self.diversity_threshold:
                self.enhance_diversity(bounds)
            self.regroup_swarms(bounds)
            self.w = max(0.2, self.w * 0.99)
            self.c1 = max(1.0, self.c1 * 0.99)
            self.c2 = min(2.5, self.c2 * 1.01)
        best_swarm_index = np.argmin(self.global_best_values)
        return self.global_best_positions[best_swarm_index]