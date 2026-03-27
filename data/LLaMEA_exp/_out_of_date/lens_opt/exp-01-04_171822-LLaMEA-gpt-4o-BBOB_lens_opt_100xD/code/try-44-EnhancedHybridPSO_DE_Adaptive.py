import numpy as np

class EnhancedHybridPSO_DE_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, 2 * dim)
        self.sub_population_size = self.population_size // 2  # Multi-population strategy
        self.populations = [None] * 2
        self.velocities = [None] * 2
        self.personal_best = [None] * 2
        self.global_best = None
        self.personal_best_fitness = [None] * 2
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0
        self.mutation_strategy = 'rand/1/bin'  # Default mutation strategy

    def _initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        for i in range(2):
            self.populations[i] = np.random.uniform(lb, ub, (self.sub_population_size, self.dim))
            self.velocities[i] = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.sub_population_size, self.dim))
            self.personal_best[i] = np.copy(self.populations[i])
            self.personal_best_fitness[i] = np.full(self.sub_population_size, np.inf)

    def _evaluate_population(self, func):
        for i in range(2):
            fitness = np.apply_along_axis(func, 1, self.populations[i])
            self.fitness_evaluations += len(fitness)
            for j in range(self.sub_population_size):
                if fitness[j] < self.personal_best_fitness[i][j]:
                    self.personal_best_fitness[i][j] = fitness[j]
                    self.personal_best[i][j] = self.populations[i][j].copy()
                    if fitness[j] < self.global_best_fitness:
                        self.global_best_fitness = fitness[j]
                        self.global_best = self.populations[i][j].copy()

    def _update_particles(self, iteration, max_iterations):
        inertia_weight = 0.4 + 0.5 * np.random.rand()
        adaptive_lr = 0.5 + 0.5 * np.random.rand()
        weighted_exploration = 0.3 * np.random.rand(self.sub_population_size, self.dim)
        for i in range(2):
            cognitive_component = 1.5 * np.random.rand(self.sub_population_size, self.dim)
            social_component = 1.5 * np.random.rand(self.sub_population_size, self.dim)
            self.velocities[i] = (adaptive_lr * inertia_weight * self.velocities[i] +
                                  cognitive_component * (self.personal_best[i] - self.populations[i]) +
                                  social_component * (self.global_best - self.populations[i]) +
                                  weighted_exploration)
            self.populations[i] += self.velocities[i]
            self.populations[i] = np.clip(self.populations[i], func.bounds.lb, func.bounds.ub)

    def _adaptive_mutation(self, bounds, func, iteration, max_iterations):
        lb, ub = bounds.lb, bounds.ub
        for pop_idx in range(2):
            for i in range(self.sub_population_size):
                if np.random.rand() < 0.2:
                    F = 0.5 + 0.5 * np.random.rand()
                    CR = 0.7 + 0.3 * np.random.rand()
                else:
                    F = 0.9
                    CR = 1.0
                indices = [idx for idx in range(self.sub_population_size) if idx != i]
                a, b, c = self.populations[pop_idx][np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(cross_points, mutant_vector, self.populations[pop_idx][i])

                fitness_trial = func(trial_vector)
                self.fitness_evaluations += 1
                if fitness_trial < self.personal_best_fitness[pop_idx][i]:
                    self.personal_best_fitness[pop_idx][i] = fitness_trial
                    self.personal_best[pop_idx][i] = trial_vector
                    if fitness_trial < self.global_best_fitness:
                        self.global_best_fitness = fitness_trial
                        self.global_best = trial_vector

    def __call__(self, func):
        bounds = func.bounds
        self._initialize_population(bounds)
        self._evaluate_population(func)
        max_iterations = self.budget // self.population_size

        for iteration in range(max_iterations):
            self._update_particles(iteration, max_iterations)
            self._evaluate_population(func)
            self._adaptive_mutation(bounds, func, iteration, max_iterations)

        return self.global_best