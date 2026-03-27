import numpy as np

class AdaptiveHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = max(5, 2 * dim)
        self.population_size = self.initial_population_size
        self.population = None
        self.velocities = None
        self.personal_best = None
        self.global_best = None
        self.personal_best_fitness = None
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0
        self.evolution_step = 0

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
        inertia_weight = 0.5 + np.random.rand() / 2
        adaptive_lr = 0.5 + np.random.rand() / 2
        cognitive_component = 2.0 * np.random.rand(self.population_size, self.dim)
        social_component = 2.0 * np.random.rand(self.population_size, self.dim)

        self.velocities = (adaptive_lr * inertia_weight * self.velocities +
                           cognitive_component * (self.personal_best - self.population) +
                           social_component * (self.global_best - self.population))
        self.population += self.velocities

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

    def _adapt_population_size(self):
        self.evolution_step += 1
        if self.evolution_step % 5 == 0:  # Every 5 iterations
            improvement_rate = np.mean(self.personal_best_fitness) / self.global_best_fitness
            if improvement_rate < 0.95:  # Adjust if progress slows
                self.population_size = min(self.population_size + 5, 4 * self.dim)
            elif improvement_rate > 0.99:
                self.population_size = max(self.initial_population_size, self.population_size - 5)

            # Reinitialize extra population if size increased
            if self.population_size > len(self.population):
                lb, ub = self.population[0].bounds.lb, self.population[0].bounds.ub
                extra_individuals = np.random.uniform(lb, ub, (self.population_size - len(self.population), self.dim))
                self.population = np.vstack((self.population, extra_individuals))
                extra_velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.population_size - len(self.velocities), self.dim))
                self.velocities = np.vstack((self.velocities, extra_velocities))
                self.personal_best = np.vstack((self.personal_best, extra_individuals))
                self.personal_best_fitness = np.append(self.personal_best_fitness, np.full(self.population_size - len(self.personal_best_fitness), np.inf))

    def __call__(self, func):
        bounds = func.bounds
        self._initialize_population(bounds)
        self._evaluate_population(func)

        while self.fitness_evaluations < self.budget:
            self._update_particles()
            self._evaluate_population(func)
            self._differential_evolution(bounds, func)
            self._adapt_population_size()
        
        return self.global_best