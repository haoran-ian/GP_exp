import numpy as np

class EnhancedAdaptivePSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, 2 * dim)
        self.sub_population_size = self.population_size // 2
        self.populations = [None] * 2
        self.velocities = [None] * 2
        self.personal_best = [None] * 2
        self.global_best = None
        self.personal_best_fitness = [None] * 2
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0
        self.dynamic_switch_threshold = self.budget // 4

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

    def _update_particles(self, adaptive_lr, l_growth_factor, topology_factor):
        inertia_weight = 0.9 - 0.5 * (self.fitness_evaluations / self.budget)
        for i in range(2):
            cognitive_component = 2.0 * np.random.rand(self.sub_population_size, self.dim)
            social_component = 2.0 * np.random.rand(self.sub_population_size, self.dim)
            
            # Topology adaptive switch: changes between global and local best influence
            if np.random.rand() < topology_factor:
                self.velocities[i] = (adaptive_lr * inertia_weight * self.velocities[i] +
                                      cognitive_component * (self.personal_best[i] - self.populations[i]) +
                                      social_component * (self.global_best - self.populations[i]))
            else:
                neighbor_best = self._get_neighbor_best(i)
                self.velocities[i] = (adaptive_lr * inertia_weight * self.velocities[i] +
                                      cognitive_component * (self.personal_best[i] - self.populations[i]) +
                                      social_component * (neighbor_best - self.populations[i]))

            self.populations[i] += self.velocities[i]

    def _get_neighbor_best(self, pop_idx):
        # Get the best neighbor within the subpopulation
        best_neighbor = np.zeros_like(self.personal_best[pop_idx])
        for i in range(self.sub_population_size):
            neighborhood_indices = np.random.choice(self.sub_population_size, 3, replace=False)
            best_idx = np.argmin(self.personal_best_fitness[pop_idx][neighborhood_indices])
            best_neighbor[i] = self.populations[pop_idx][neighborhood_indices[best_idx]]
        return best_neighbor

    def _differential_evolution(self, bounds, func):
        lb, ub = bounds.lb, bounds.ub
        for pop_idx in range(2):
            for i in range(self.sub_population_size):
                F = 0.6 + np.random.rand() * 0.4
                CR = 0.7 + np.random.rand() * 0.3
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

        while self.fitness_evaluations < self.budget:
            chaotic_sequence = 0.8 * np.sin(np.pi * self.fitness_evaluations / self.budget)
            adaptive_lr = 0.6 + (self.fitness_evaluations / self.budget) * chaotic_sequence
            l_growth_factor = 0.25 * (1 - (self.fitness_evaluations / self.budget))
            topology_factor = 0.5 * (1 - chaotic_sequence)  # New factor for topology adaptation

            self._update_particles(adaptive_lr, l_growth_factor, topology_factor)
            self._evaluate_population(func)
            if self.fitness_evaluations > self.dynamic_switch_threshold:
                self._differential_evolution(bounds, func)
        
        return self.global_best