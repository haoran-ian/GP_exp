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
        self.initial_constriction_factor = 0.9
        self.final_constriction_factor = 0.4
        self.cognitive_constant = 2.05
        self.social_constant = 2.05
        self.differential_weight = 0.8
        self.crossover_rate = 0.9
        self.iteration = 0

    def evaluate_population(self, func):
        for i in range(self.population_size):
            value = func(self.population[i])
            if value < self.fitness[i]:
                self.fitness[i] = value
                self.personal_best[i] = self.population[i].copy()
            if self.global_best is None or value < func(self.global_best):
                self.global_best = self.population[i].copy()

    def adaptive_parameters(self):
        progress = self.iteration / (self.budget / self.population_size)
        self.constriction_factor = self.initial_constriction_factor - progress * (self.initial_constriction_factor - self.final_constriction_factor)
        self.cognitive_constant = 1.5 + 2.5 * np.sin(progress * np.pi)
        self.social_constant = 1.5 + 2.5 * np.cos(progress * np.pi)
    
    def chaos_sequence(self, size):
        x = np.random.rand()
        for _ in range(size):
            x = 4.0 * x * (1.0 - x)
            yield x
    
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
            
    def __call__(self, func):
        self.population = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * self.population
        for _ in range(0, self.budget, self.population_size):
            self.evaluate_population(func)
            self.adaptive_parameters()
            self.differential_evolution_step(func)
            self.particle_swarm_step(func)
            self.iteration += 1
        return self.global_best