import numpy as np

class AdaptiveChaoticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = None
        self.fitness = None
        self.chaos_sequence = self.generate_chaos_sequence(budget)

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
                self.fitness[i] = func(individual)
    
    def adaptive_chaotic_differential_evolution(self, func, chaos_index):
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[indices]
            chaotic_factor = 0.5 + 0.5 * self.chaos_sequence[chaos_index]  # Self-adaptive chaotic scaling
            mutant = np.clip(a + self.mutation_factor * chaotic_factor * (b - c), func.bounds.lb, func.bounds.ub)
            cross_points = np.random.rand(self.dim) < self.crossover_rate
            trial = np.where(cross_points, mutant, self.population[i])
            trial_fitness = func(trial)
            if trial_fitness < self.fitness[i]:
                self.population[i], self.fitness[i] = trial, trial_fitness

    def dynamic_population_resizing(self, generation):
        if generation % 5 == 0:
            self.population_size = max(5, int(self.population_size * 0.9))
            self.population = self.population[:self.population_size]
            self.fitness = self.fitness[:self.population_size]

    def local_search(self, individual, func, step_size=0.1):
        direction = np.random.uniform(-1, 1, self.dim)
        neighbor = np.clip(individual + step_size * direction, func.bounds.lb, func.bounds.ub)
        neighbor_fitness = func(neighbor)
        if neighbor_fitness < func(individual):
            return neighbor, neighbor_fitness
        return individual, func(individual)

    def __call__(self, func):
        self.initialize_population(func.bounds.lb, func.bounds.ub)
        self.evaluate_population(func)
        evaluations = self.population_size
        chaos_index = 0
        generation = 1

        while evaluations < self.budget:
            self.adaptive_chaotic_differential_evolution(func, chaos_index)
            evaluations += self.population_size
            chaos_index = min(chaos_index + self.population_size, len(self.chaos_sequence) - 1)
            
            if evaluations < self.budget:
                for i in range(self.population_size):
                    self.population[i], self.fitness[i] = self.local_search(self.population[i], func)
                    evaluations += 1
                    if evaluations >= self.budget:
                        break

            self.dynamic_population_resizing(generation)
            generation += 1

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]