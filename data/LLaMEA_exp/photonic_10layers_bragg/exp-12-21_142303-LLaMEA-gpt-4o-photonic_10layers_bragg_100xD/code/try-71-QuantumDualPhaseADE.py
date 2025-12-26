import numpy as np

class QuantumDualPhaseADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.q_population = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.elite_count = 2
        self.phase_switch_interval = 50
        self.diversity_boost_interval = 100
        self.dynamic_adaptation = True
        self.exploration_phase = True

    def _quantum_observation(self):
        angles = np.arccos(1 - 2 * self.q_population)
        return (np.cos(angles) > np.random.rand(*angles.shape)).astype(float)

    def _evaluate_population(self, func, bounds):
        real_population = bounds.lb + self._quantum_observation() * (bounds.ub - bounds.lb)
        fitness = np.array([func(ind) for ind in real_population])
        return real_population, fitness

    def _update_best(self, real_population, fitness):
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_fitness:
            self.best_fitness = fitness[min_idx]
            self.best_solution = real_population[min_idx]

    def _adaptive_mutation(self, diversity):
        mutation_strength = np.abs(np.random.normal(0, diversity * (self.best_fitness / 10), self.q_population.shape))
        adapt_factor = np.random.rand(*self.q_population.shape)
        self.q_population += mutation_strength * (adapt_factor - 0.5)
        self.q_population = np.clip(self.q_population, 0, 1)

    def _differential_evolution(self, real_population, bounds, diversity):
        trial_population = np.copy(real_population)
        for i in range(self.elite_count, self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = real_population[a] + self.mutation_factor * (real_population[b] - real_population[c])
            mutant = np.clip(mutant, bounds.lb, bounds.ub)
            crossover = np.random.rand(self.dim) < (self.crossover_rate + diversity)
            trial_population[i] = np.where(crossover, mutant, real_population[i])
        return trial_population

    def _calculate_diversity(self, real_population):
        centroid = np.mean(real_population, axis=0)
        diversity = np.mean(np.linalg.norm(real_population - centroid, axis=1))
        return diversity / self.dim

    def _adaptive_rates(self, evaluations, diversity):
        if self.dynamic_adaptation:
            progress = evaluations / self.budget
            if diversity < 0.2:
                self.mutation_factor = min(0.7, 0.5 + 0.2 * progress)
            else:
                self.mutation_factor = max(0.3, 0.5 - 0.2 * progress)
            self.crossover_rate = min(0.9, 0.7 + 0.2 * (1 - diversity))

    def __call__(self, func):
        bounds = func.bounds
        evaluations = 0

        while evaluations < self.budget:
            real_population, fitness = self._evaluate_population(func, bounds)
            self._update_best(real_population, fitness)
            evaluations += self.population_size

            diversity = self._calculate_diversity(real_population)
            self._adaptive_rates(evaluations, diversity)
            
            if self.exploration_phase:
                trial_population = self._quantum_observation() * (bounds.ub - bounds.lb) + bounds.lb
            else:
                trial_population = self._differential_evolution(real_population, bounds, diversity)
            
            trial_fitness = np.array([func(ind) for ind in trial_population])

            for i in range(self.elite_count, self.population_size):
                if trial_fitness[i] < fitness[i]:
                    fitness[i] = trial_fitness[i]
                    real_population[i] = trial_population[i]

            if evaluations % self.diversity_boost_interval == 0:
                self.q_population[:self.elite_count] = np.random.uniform(0, 1, (self.elite_count, self.dim))
                self.q_population[self.elite_count:self.population_size // 2] = np.random.uniform(0, 1, (self.population_size // 2 - self.elite_count, self.dim))

            self._adaptive_mutation(diversity)

            if evaluations % self.phase_switch_interval == 0:
                self.exploration_phase = not self.exploration_phase

        return self.best_solution