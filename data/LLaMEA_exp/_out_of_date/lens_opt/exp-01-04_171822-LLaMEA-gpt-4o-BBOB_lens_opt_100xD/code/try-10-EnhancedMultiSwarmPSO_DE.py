import numpy as np

class EnhancedMultiSwarmPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, 2 * dim)
        self.swarm_count = 3  # Number of swarms
        self.swarm_sizes = [self.population_size // self.swarm_count for _ in range(self.swarm_count)]
        self.swarm_sizes[-1] += self.population_size % self.swarm_count
        self.populations = [None] * self.swarm_count
        self.velocities = [None] * self.swarm_count
        self.personal_best = [None] * self.swarm_count
        self.global_bests = [None] * self.swarm_count
        self.personal_best_fitness = [None] * self.swarm_count
        self.global_best_fitness = np.full(self.swarm_count, np.inf)
        self.fitness_evaluations = 0

    def _initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        for i in range(self.swarm_count):
            self.populations[i] = np.random.uniform(lb, ub, (self.swarm_sizes[i], self.dim))
            self.velocities[i] = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.swarm_sizes[i], self.dim))
            self.personal_best[i] = np.copy(self.populations[i])
            self.personal_best_fitness[i] = np.full(self.swarm_sizes[i], np.inf)
            self.global_bests[i] = np.copy(self.populations[i][0])

    def _evaluate_population(self, func, swarm_idx):
        fitness = np.apply_along_axis(func, 1, self.populations[swarm_idx])
        self.fitness_evaluations += len(fitness)
        for i in range(self.swarm_sizes[swarm_idx]):
            if fitness[i] < self.personal_best_fitness[swarm_idx][i]:
                self.personal_best_fitness[swarm_idx][i] = fitness[i]
                self.personal_best[swarm_idx][i] = self.populations[swarm_idx][i].copy()
                if fitness[i] < self.global_best_fitness[swarm_idx]:
                    self.global_best_fitness[swarm_idx] = fitness[i]
                    self.global_bests[swarm_idx] = self.populations[swarm_idx][i].copy()

    def _update_particles(self, swarm_idx):
        inertia_weight = 0.5 + np.random.rand() / 2
        adaptive_lr = 0.5 + np.random.rand() / 2
        cognitive_component = 2.0 * np.random.rand(self.swarm_sizes[swarm_idx], self.dim)
        social_component = 2.0 * np.random.rand(self.swarm_sizes[swarm_idx], self.dim)

        self.velocities[swarm_idx] = (adaptive_lr * inertia_weight * self.velocities[swarm_idx] +
                                      cognitive_component * (self.personal_best[swarm_idx] - self.populations[swarm_idx]) +
                                      social_component * (self.global_bests[swarm_idx] - self.populations[swarm_idx]))
        self.populations[swarm_idx] += self.velocities[swarm_idx]

    def _differential_evolution(self, bounds, func, swarm_idx):
        lb, ub = bounds.lb, bounds.ub
        F = 0.5 + np.random.rand() * 0.5
        CR = 0.7 + np.random.rand() * 0.3

        for i in range(self.swarm_sizes[swarm_idx]):
            indices = [idx for idx in range(self.swarm_sizes[swarm_idx]) if idx != i]
            a, b, c = self.populations[swarm_idx][np.random.choice(indices, 3, replace=False)]
            mutant_vector = np.clip(a + F * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial_vector = np.where(cross_points, mutant_vector, self.populations[swarm_idx][i])

            fitness_trial = func(trial_vector)
            self.fitness_evaluations += 1
            if fitness_trial < self.personal_best_fitness[swarm_idx][i]:
                self.personal_best_fitness[swarm_idx][i] = fitness_trial
                self.personal_best[swarm_idx][i] = trial_vector
                if fitness_trial < self.global_best_fitness[swarm_idx]:
                    self.global_best_fitness[swarm_idx] = fitness_trial
                    self.global_bests[swarm_idx] = trial_vector

    def __call__(self, func):
        bounds = func.bounds
        self._initialize_population(bounds)

        for swarm_idx in range(self.swarm_count):
            self._evaluate_population(func, swarm_idx)

        while self.fitness_evaluations < self.budget:
            for swarm_idx in range(self.swarm_count):
                self._update_particles(swarm_idx)
                self._evaluate_population(func, swarm_idx)
                self._differential_evolution(bounds, func, swarm_idx)

        best_swarm_idx = np.argmin(self.global_best_fitness)
        return self.global_bests[best_swarm_idx]