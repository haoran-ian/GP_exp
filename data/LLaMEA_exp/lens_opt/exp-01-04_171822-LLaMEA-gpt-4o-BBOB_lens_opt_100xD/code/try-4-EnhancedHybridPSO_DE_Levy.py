import numpy as np

class EnhancedHybridPSO_DE_Levy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, 2 * dim)
        self.population = None
        self.velocities = None
        self.personal_best = None
        self.global_best = None
        self.personal_best_fitness = None
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0

    def _initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.population_size, self.dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_fitness = np.full(self.population_size, np.inf)

    def _evaluate_population(self, func):
        fitness = np.apply_along_axis(func, 1, self.population)
        self.fitness_evaluations += len(fitness)
        for i in range(self.population_size):
            if fitness[i] < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness[i]
                self.personal_best[i] = self.population[i].copy()
                if fitness[i] < self.global_best_fitness:
                    self.global_best_fitness = fitness[i]
                    self.global_best = self.population[i].copy()

    def _update_particles(self):
        inertia_weight = 0.5 + np.random.rand() / 2  # Adaptive inertia weight
        cognitive_component = 2.0 * np.random.rand(self.population_size, self.dim)
        social_component = 2.0 * np.random.rand(self.population_size, self.dim)

        self.velocities = (inertia_weight * self.velocities +
                           cognitive_component * (self.personal_best - self.population) +
                           social_component * (self.global_best - self.population))
        self.population += self.velocities

    def _levy_flight(self, size):
        beta = 3/2
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.randn(size) * sigma
        v = np.random.randn(size)
        step = u / abs(v)**(1 / beta)
        return step

    def _differential_evolution(self, bounds, func):
        lb, ub = bounds.lb, bounds.ub
        F = 0.5 + np.random.rand() * 0.5
        CR = 0.7 + np.random.rand() * 0.3

        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            mutant_vector = np.clip(a + F * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial_vector = np.where(cross_points, mutant_vector, self.population[i])

            fitness_trial = func(trial_vector)
            self.fitness_evaluations += 1
            if fitness_trial < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness_trial
                self.personal_best[i] = trial_vector
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

            # Introduce Levy flights for exploration
            if self.fitness_evaluations < self.budget:
                levy_step = self._levy_flight(self.dim)
                new_position = self.global_best + levy_step
                new_position = np.clip(new_position, bounds.lb, bounds.ub)
                new_fitness = func(new_position)
                self.fitness_evaluations += 1
                if new_fitness < self.global_best_fitness:
                    self.global_best_fitness = new_fitness
                    self.global_best = new_position

        return self.global_best