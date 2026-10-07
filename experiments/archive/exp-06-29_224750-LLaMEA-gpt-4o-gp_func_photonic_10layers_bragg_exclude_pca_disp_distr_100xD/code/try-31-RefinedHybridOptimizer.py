import numpy as np

class RefinedHybridOptimizer:
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
        self.initial_differential_weight = 0.5
        self.final_differential_weight = 0.7
        self.crossover_rate = 0.9
        self.elite_ratio = 0.1

    def evaluate_population(self, func):
        for i in range(self.population_size):
            value = func(self.population[i])
            if value < self.fitness[i]:
                self.fitness[i] = value
                self.personal_best[i] = self.population[i].copy()
            if self.global_best is None or value < func(self.global_best):
                self.global_best = self.population[i].copy()

    def adaptive_differential_evolution_step(self, func, generation):
        diff_weight = self.initial_differential_weight + (
            (self.final_differential_weight - self.initial_differential_weight) * 
            (generation / (self.budget // self.population_size))
        )
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[indices]
            mutant_vector = a + diff_weight * (b - c)
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

    def elitism_selection(self):
        elite_count = int(self.elite_ratio * self.population_size)
        elite_indices = np.argsort(self.fitness)[:elite_count]
        elite_population = self.population[elite_indices]
        return elite_population

    def __call__(self, func):
        self.population = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * self.population
        generation = 0
        while self.budget > 0:
            self.evaluate_population(func)
            elites = self.elitism_selection()
            self.adaptive_differential_evolution_step(func, generation)
            self.particle_swarm_step(func)
            self.population[:len(elites)] = elites  # Preserve elites
            generation += 1
            self.budget -= self.population_size
        return self.global_best