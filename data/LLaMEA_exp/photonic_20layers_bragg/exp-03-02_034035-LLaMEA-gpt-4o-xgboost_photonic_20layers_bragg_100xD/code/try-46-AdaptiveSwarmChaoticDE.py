import numpy as np

class AdaptiveSwarmChaoticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.base_mutation_factor = 0.8
        self.base_crossover_rate = 0.9
        self.inertia_weight = 0.5
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.population = None
        self.velocity = None
        self.fitness = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.global_best = None
        self.global_best_fitness = np.inf
        self.chaos_sequence = self.generate_chaos_sequence(budget)

    def generate_chaos_sequence(self, size):
        chaos_sequence = np.zeros(size)
        chaos_sequence[0] = np.random.rand()
        for i in range(1, size):
            chaos_sequence[i] = 4.0 * chaos_sequence[i-1] * (1.0 - chaos_sequence[i-1])
        return chaos_sequence

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(low=lb, high=ub, size=(self.population_size, self.dim))
        self.velocity = np.random.uniform(low=-1, high=1, size=(self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i, individual in enumerate(self.population):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(individual)
                if self.fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best[i], self.personal_best_fitness[i] = individual, self.fitness[i]
                if self.fitness[i] < self.global_best_fitness:
                    self.global_best, self.global_best_fitness = individual, self.fitness[i]

    def chaotic_differential_evolution(self, func, chaos_index):
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[indices]
            chaotic_factor = self.chaos_sequence[chaos_index]
            adaptive_mutation_factor = self.base_mutation_factor * (0.5 + chaotic_factor)
            adaptive_crossover_rate = self.base_crossover_rate * (0.5 + chaotic_factor)
            mutant = np.clip(a + adaptive_mutation_factor * (b - c), func.bounds.lb, func.bounds.ub)
            cross_points = np.random.rand(self.dim) < adaptive_crossover_rate
            trial = np.where(cross_points, mutant, self.population[i])
            trial_fitness = func(trial)
            if trial_fitness < self.fitness[i]:
                self.population[i], self.fitness[i] = trial, trial_fitness
                if trial_fitness < self.personal_best_fitness[i]:
                    self.personal_best[i], self.personal_best_fitness[i] = trial, trial_fitness
                if trial_fitness < self.global_best_fitness:
                    self.global_best, self.global_best_fitness = trial, trial_fitness

    def update_velocity_and_position(self, func):
        for i in range(self.population_size):
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive_component = self.cognitive_weight * r1 * (self.personal_best[i] - self.population[i])
            social_component = self.social_weight * r2 * (self.global_best - self.population[i])
            self.velocity[i] = (self.inertia_weight * self.velocity[i] +
                                cognitive_component + social_component)
            self.population[i] = np.clip(self.population[i] + self.velocity[i], func.bounds.lb, func.bounds.ub)

    def __call__(self, func):
        self.initialize_population(func.bounds.lb, func.bounds.ub)
        self.evaluate_population(func)
        evaluations = self.population_size
        chaos_index = 0

        while evaluations < self.budget:
            self.chaotic_differential_evolution(func, chaos_index)
            evaluations += self.population_size
            chaos_index = min(chaos_index + self.population_size, len(self.chaos_sequence) - 1)
            
            if evaluations < self.budget:
                self.update_velocity_and_position(func)
                self.evaluate_population(func)
                evaluations += self.population_size

        return self.global_best, self.global_best_fitness