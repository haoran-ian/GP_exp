import numpy as np

class MultiChaosDEwithAdaptiveRates:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = None
        self.fitness = None
        self.chaos_sequences = self.generate_chaos_sequences(budget, 3)

    def generate_chaos_sequences(self, size, count):
        chaos_sequences = np.zeros((count, size))
        for j in range(count):
            chaos_sequences[j][0] = np.random.rand()
            for i in range(1, size):
                chaos_sequences[j][i] = 4.0 * chaos_sequences[j][i-1] * (1.0 - chaos_sequences[j][i-1])
        return chaos_sequences

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(low=lb, high=ub, size=(self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i, individual in enumerate(self.population):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(individual)
    
    def chaotic_differential_evolution(self, func, chaos_index):
        diversity = np.mean(np.std(self.population, axis=0))
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[indices]
            
            chaotic_factor_m = 1 + self.chaos_sequences[0][chaos_index]
            chaotic_factor_c = 1 + self.chaos_sequences[1][chaos_index]
            adaptive_mutation_factor = self.mutation_factor * diversity * chaotic_factor_m * (1 - chaos_index/self.budget)
            adaptive_crossover_rate = self.crossover_rate * chaotic_factor_c
            
            mutant = np.clip(a + adaptive_mutation_factor * (b - c), func.bounds.lb, func.bounds.ub)
            cross_points = np.random.rand(self.dim) < adaptive_crossover_rate
            trial = np.where(cross_points, mutant, self.population[i])
            trial_fitness = func(trial)
            if trial_fitness < self.fitness[i]:
                self.population[i], self.fitness[i] = trial, trial_fitness

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

        while evaluations < self.budget:
            self.chaotic_differential_evolution(func, chaos_index)
            evaluations += self.population_size
            chaos_index = min(chaos_index + self.population_size, self.budget - 1)
            
            if evaluations < self.budget:
                for i in range(self.population_size):
                    self.population[i], self.fitness[i] = self.local_search(self.population[i], func)
                    evaluations += 1
                    if evaluations >= self.budget:
                        break

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]