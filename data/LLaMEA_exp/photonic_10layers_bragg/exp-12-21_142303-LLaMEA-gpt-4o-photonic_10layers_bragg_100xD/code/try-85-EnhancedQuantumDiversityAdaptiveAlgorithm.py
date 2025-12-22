import numpy as np

class EnhancedQuantumDiversityAdaptiveAlgorithm:
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
        self.diversity_boost_interval = 50
        self.dynamic_adaptation = True
        self.learning_automata_rate = 0.1

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

    def _entropy_based_diversity(self, real_population):
        normalized_pop = (real_population - np.min(real_population, axis=0)) / (np.max(real_population, axis=0) - np.min(real_population, axis=0) + 1e-9)
        entropy = -np.sum(normalized_pop * np.log(normalized_pop + 1e-9), axis=0)
        return np.mean(entropy)

    def _adaptive_mutation(self, entropy_diversity):
        mutation_strength = np.abs(np.random.normal(0, entropy_diversity * (self.best_fitness / 10), self.q_population.shape))
        adapt_factor = np.random.rand(*self.q_population.shape)
        self.q_population += mutation_strength * (adapt_factor - 0.5)
        self.q_population = np.clip(self.q_population, 0, 1)

    def _differential_evolution(self, real_population, bounds, entropy_diversity):
        trial_population = np.copy(real_population)
        for i in range(self.elite_count, self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = real_population[a] + self.mutation_factor * (real_population[b] - real_population[c])
            mutant = np.clip(mutant, bounds.lb, bounds.ub)
            crossover = np.random.rand(self.dim) < (self.crossover_rate + 0.5 * entropy_diversity)
            trial_population[i] = np.where(crossover, mutant, real_population[i])
        return trial_population

    def _adaptive_rates(self, evaluations, entropy_diversity):
        if self.dynamic_adaptation:
            progress = evaluations / self.budget
            self.mutation_factor = 0.5 + 0.3 * (1 - entropy_diversity)
            self.crossover_rate = 0.7 + 0.2 * entropy_diversity

            # Learning automata for adaptive adjustment
            if np.random.rand() < self.learning_automata_rate:
                reward = 1.0 if entropy_diversity < 0.3 else -1.0
                self.mutation_factor = np.clip(self.mutation_factor + reward * 0.05, 0.2, 0.8)
                self.crossover_rate = np.clip(self.crossover_rate + reward * 0.03, 0.4, 0.9)

    def __call__(self, func):
        bounds = func.bounds
        evaluations = 0

        while evaluations < self.budget:
            real_population, fitness = self._evaluate_population(func, bounds)
            self._update_best(real_population, fitness)
            evaluations += self.population_size

            entropy_diversity = self._entropy_based_diversity(real_population)
            self._adaptive_rates(evaluations, entropy_diversity)
            trial_population = self._differential_evolution(real_population, bounds, entropy_diversity)
            trial_fitness = np.array([func(ind) for ind in trial_population])

            for i in range(self.elite_count, self.population_size):
                if trial_fitness[i] < fitness[i]:
                    fitness[i] = trial_fitness[i]
                    real_population[i] = trial_population[i]

            if evaluations % self.diversity_boost_interval == 0:
                self.q_population[:self.elite_count] = np.random.uniform(0, 1, (self.elite_count, self.dim))
                self.q_population[self.elite_count:self.population_size // 2] = np.random.uniform(0, 1, (self.population_size // 2 - self.elite_count, self.dim))

            self._adaptive_mutation(entropy_diversity)

        return self.best_solution