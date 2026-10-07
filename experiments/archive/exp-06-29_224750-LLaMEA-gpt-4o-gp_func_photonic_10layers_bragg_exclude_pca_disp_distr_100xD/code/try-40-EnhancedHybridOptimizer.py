import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.rand(self.population_size, dim)
        self.velocities = np.zeros((self.population_size, dim))
        self.personal_best = self.population.copy()
        self.global_best = None
        self.fitness = np.full(self.population_size, np.inf)
        self.constriction_factor = 0.729
        self.cognitive_constant = 1.49445
        self.social_constant = 1.49445
        self.differential_weight = 0.8
        self.crossover_rate = 0.9
        self.elite_fraction = 0.1

    def evaluate_population(self, func):
        for i in range(self.population_size):
            value = func(self.population[i])
            if value < self.fitness[i]:
                self.fitness[i] = value
                self.personal_best[i] = self.population[i].copy()
            if self.global_best is None or value < func(self.global_best):
                self.global_best = self.population[i].copy()

    def differential_evolution_step(self, func):
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[indices]
            mutant_vector = a + self.differential_weight * (b - c)
            crossover = np.random.rand(self.dim) < self.crossover_rate
            trial_vector = np.where(crossover, mutant_vector, self.population[i])
            trial_vector = np.clip(trial_vector, func.bounds.lb, func.bounds.ub)
            trial_value = func(trial_vector)
            if trial_value < self.fitness[i]:
                self.population[i] = trial_vector
                self.fitness[i] = trial_value
                self.personal_best[i] = trial_vector
                if trial_value < func(self.global_best):
                    self.global_best = trial_vector

    def particle_swarm_step(self, func):
        for i in range(self.population_size):
            cognitive_component = self.cognitive_constant * np.random.rand(self.dim) * (self.personal_best[i] - self.population[i])
            social_component = self.social_constant * np.random.rand(self.dim) * (self.global_best - self.population[i])
            self.velocities[i] = (self.constriction_factor * (self.velocities[i] + cognitive_component + social_component))
            self.population[i] = np.clip(self.population[i] + self.velocities[i], func.bounds.lb, func.bounds.ub)

    def elitism(self):
        elite_size = int(self.elite_fraction * self.population_size)
        elite_indices = np.argsort(self.fitness)[:elite_size]
        elite_population = self.population[elite_indices].copy()
        return elite_population

    def adaptive_tuning(self, iteration, max_iterations):
        # Linearly decrease the differential weight for exploration-exploitation balance
        self.differential_weight = 0.8 - 0.5 * (iteration / max_iterations)
        # Increase the crossover rate over time for more diversity
        self.crossover_rate = 0.7 + 0.3 * (iteration / max_iterations)

    def __call__(self, func):
        self.population = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * self.population
        max_iterations = self.budget // self.population_size
        for iteration in range(max_iterations):
            self.evaluate_population(func)
            self.adaptive_tuning(iteration, max_iterations)
            elite_population = self.elitism()
            self.differential_evolution_step(func)
            self.particle_swarm_step(func)
            # Integrate elite solutions back into the population
            self.population[:len(elite_population)] = elite_population
        return self.global_best