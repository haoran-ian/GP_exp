import numpy as np

class EnhancedMultiPopPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_subpopulations = 3
        self.subpop_size = max(5, 2 * dim // self.num_subpopulations)
        self.subpopulations = [None] * self.num_subpopulations
        self.velocities = [None] * self.num_subpopulations
        self.personal_best = [None] * self.num_subpopulations
        self.global_best = None
        self.personal_best_fitness = [None] * self.num_subpopulations
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0

    def _initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        for i in range(self.num_subpopulations):
            self.subpopulations[i] = np.random.uniform(lb, ub, (self.subpop_size, self.dim))
            self.velocities[i] = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.subpop_size, self.dim))
            self.personal_best[i] = np.copy(self.subpopulations[i])
            self.personal_best_fitness[i] = np.full(self.subpop_size, np.inf)

    def _evaluate_population(self, func):
        for i in range(self.num_subpopulations):
            fitness = np.apply_along_axis(func, 1, self.subpopulations[i])
            self.fitness_evaluations += len(fitness)
            for j in range(self.subpop_size):
                if fitness[j] < self.personal_best_fitness[i][j]:
                    self.personal_best_fitness[i][j] = fitness[j]
                    self.personal_best[i][j] = self.subpopulations[i][j].copy()
                    if fitness[j] < self.global_best_fitness:
                        self.global_best_fitness = fitness[j]
                        self.global_best = self.subpopulations[i][j].copy()

    def _update_particles(self):
        for i in range(self.num_subpopulations):
            inertia_weight = 0.5 + np.random.rand() / 2
            adaptive_lr = 0.5 + np.random.rand() / 2
            cognitive_component = 2.0 * np.random.rand(self.subpop_size, self.dim)
            social_component = 2.0 * np.random.rand(self.subpop_size, self.dim)

            self.velocities[i] = (adaptive_lr * inertia_weight * self.velocities[i] +
                                  cognitive_component * (self.personal_best[i] - self.subpopulations[i]) +
                                  social_component * (self.global_best - self.subpopulations[i]))
            self.subpopulations[i] += self.velocities[i]

    def _differential_evolution(self, bounds, func):
        lb, ub = bounds.lb, bounds.ub
        for i in range(self.num_subpopulations):
            F = 0.5 + np.random.rand() * 0.5
            CR = 0.7 + np.random.rand() * 0.3

            for j in range(self.subpop_size):
                indices = [idx for idx in range(self.subpop_size) if idx != j]
                a, b, c = self.subpopulations[i][np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(cross_points, mutant_vector, self.subpopulations[i][j])

                fitness_trial = func(trial_vector)
                self.fitness_evaluations += 1
                if fitness_trial < self.personal_best_fitness[i][j]:
                    self.personal_best_fitness[i][j] = fitness_trial
                    self.personal_best[i][j] = trial_vector
                    if fitness_trial < self.global_best_fitness:
                        self.global_best_fitness = fitness_trial
                        self.global_best = trial_vector

    def __call__(self, func):
        bounds = func.bounds
        self._initialize_population(bounds)
        self._evaluate_population(func)

        while self.fitness_evaluations < self.budget:
            self._update_particles()
            self._evaluate_population(func)
            self._differential_evolution(bounds, func)
        
        return self.global_best