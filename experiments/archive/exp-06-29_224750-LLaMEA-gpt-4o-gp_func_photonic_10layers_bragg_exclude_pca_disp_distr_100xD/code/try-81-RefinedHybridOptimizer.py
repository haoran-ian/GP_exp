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
        self.differential_weight = 0.8
        self.crossover_rate = 0.9
        self.adaptive_factor = 0.2

    def evaluate_population(self, func):
        for i in range(self.population_size):
            value = func(self.population[i])
            if value < self.fitness[i]:
                self.fitness[i] = value
                self.personal_best[i] = self.population[i].copy()
            if self.global_best is None or value < func(self.global_best):
                self.global_best = self.population[i].copy()

    def adaptive_differential_evolution_step(self, func):
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[indices]
            f_adapt = self.differential_weight + self.adaptive_factor * (0.5 - np.random.rand())
            mutant_vector = a + f_adapt * (b - c)
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

    def enhanced_particle_swarm_step(self, func):
        for i in range(self.population_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive_component = self.cognitive_constant * r1 * (self.personal_best[i] - self.population[i])
            social_component = self.social_constant * r2 * (self.global_best - self.population[i])
            inertia_weight = 0.9 - (0.5 * (self.budget - _)) / self.budget
            self.velocities[i] = (inertia_weight * self.velocities[i] + cognitive_component + social_component)
            self.population[i] = np.clip(self.population[i] + self.velocities[i], func.bounds.lb, func.bounds.ub)

    def __call__(self, func):
        self.population = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * self.population
        for _ in range(0, self.budget, self.population_size):
            self.evaluate_population(func)
            self.adaptive_differential_evolution_step(func)
            self.enhanced_particle_swarm_step(func)
        return self.global_best