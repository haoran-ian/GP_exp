import numpy as np

class SelfAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 24 * dim  # Further increased for enhanced exploration
        self.base_mutation_factor = 0.7
        self.base_crossover_rate = 0.85
        self.population = None
        self.fitness = None
        self.chaos_sequence = self.generate_chaos_sequence(budget)
        self.meta_learning_rate = 0.05
        self.mutation_factors = np.full(self.population_size, self.base_mutation_factor)

    def generate_chaos_sequence(self, size):
        chaos_sequence = np.zeros(size)
        chaos_sequence[0] = np.random.rand()
        for i in range(1, size):
            chaos_sequence[i] = 4.0 * chaos_sequence[i-1] * (1.0 - chaos_sequence[i-1])
        return chaos_sequence

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(low=lb, high=ub, size=(self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
    
    def evaluate_population(self, func):
        for i, individual in enumerate(self.population):
            if self.fitness[i] == np.inf:
                self.fitness[i] = self.transformed_fitness(func, individual)

    def transformed_fitness(self, func, individual):
        base_fitness = func(individual)
        non_uniformity = np.std(self.population, axis=0).mean()
        return base_fitness * (1 + non_uniformity)

    def chaotic_differential_evolution(self, func, chaos_index):
        exploration_factor = np.std(self.population, axis=0).mean()
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[indices]
            chaotic_factor = self.chaos_sequence[chaos_index]
            adaptive_mutation_factor = self.mutation_factors[i]
            adaptive_crossover_rate = self.base_crossover_rate * (0.5 + chaotic_factor)
            mutant = np.clip(a + adaptive_mutation_factor * (b - c), func.bounds.lb, func.bounds.ub)
            cross_points = np.random.rand(self.dim) < adaptive_crossover_rate
            trial = np.where(cross_points, mutant, self.population[i])
            trial_fitness = self.transformed_fitness(func, trial)
            if trial_fitness < self.fitness[i]:
                self.population[i], self.fitness[i] = trial, trial_fitness
                self.mutation_factors[i] *= (1 + self.meta_learning_rate)
            else:
                self.mutation_factors[i] *= (1 - self.meta_learning_rate)

    def adaptive_local_search(self, individual, fitness, func):
        step_size = 0.02
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            perturb = np.zeros(self.dim)
            perturb[i] = step_size
            gradient[i] = (self.transformed_fitness(func, individual + perturb) - self.transformed_fitness(func, individual - perturb)) / (2 * step_size)
        step_factor = min(1.0, 0.1 / np.linalg.norm(gradient))
        new_individual = np.clip(individual - step_size * gradient * step_factor, func.bounds.lb, func.bounds.ub)
        new_fitness = self.transformed_fitness(func, new_individual)
        if new_fitness < fitness:
            return new_individual, new_fitness
        return individual, fitness

    def meta_learning_adjustment(self, evaluations):
        self.base_mutation_factor += self.meta_learning_rate * np.tanh((self.budget - evaluations) / self.budget)
        self.base_crossover_rate -= self.meta_learning_rate * np.tanh((self.budget - evaluations) / self.budget)

    def __call__(self, func):
        self.initialize_population(func.bounds.lb, func.bounds.ub)
        self.evaluate_population(func)
        evaluations = self.population_size
        chaos_index = 0

        while evaluations < self.budget:
            self.meta_learning_adjustment(evaluations)
            self.chaotic_differential_evolution(func, chaos_index)
            evaluations += self.population_size
            chaos_index = min(chaos_index + self.population_size, len(self.chaos_sequence) - 1)
            
            if evaluations < self.budget:
                for i in range(self.population_size):
                    self.population[i], self.fitness[i] = self.adaptive_local_search(self.population[i], self.fitness[i], func)
                    evaluations += 1
                    if evaluations >= self.budget:
                        break

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]