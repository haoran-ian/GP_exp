import numpy as np

class EnhancedChaoticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = None
        self.fitness = None
        self.chaos_sequence = self.generate_chaos_sequence(budget)
        self.best_solution = None
        self.best_fitness = np.inf

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
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = individual
    
    def chaotic_differential_evolution(self, func, chaos_index):
        diversity = np.mean(np.std(self.population, axis=0))
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[indices]
            chaotic_factor = 1 + self.chaos_sequence[chaos_index]
            adaptive_mutation_factor = self.mutation_factor * diversity * (1 - chaos_index/self.budget)
            mutant = np.clip(a + adaptive_mutation_factor * chaotic_factor * (b - c), func.bounds.lb, func.bounds.ub)
            cross_points = np.random.rand(self.dim) < self.crossover_rate
            trial = np.where(cross_points, mutant, self.population[i])
            trial_fitness = func(trial)
            if trial_fitness < self.fitness[i]:
                self.population[i], self.fitness[i] = trial, trial_fitness
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

    def local_search(self, individual, func, step_size=0.1):
        direction = np.random.uniform(-1, 1, self.dim)
        neighbor = np.clip(individual + step_size * direction, func.bounds.lb, func.bounds.ub)
        neighbor_fitness = func(neighbor)
        if neighbor_fitness < func(individual):
            return neighbor, neighbor_fitness
        return individual, func(individual)

    def progressive_local_search(self, func, improvement_threshold=0.01):
        for i in range(self.population_size):
            local_best = self.population[i]
            local_best_fitness = self.fitness[i]
            step_size = 0.1
            while step_size > 0.001:
                neighbor, neighbor_fitness = self.local_search(local_best, func, step_size)
                if neighbor_fitness < local_best_fitness - improvement_threshold:
                    local_best, local_best_fitness = neighbor, neighbor_fitness
                else:
                    step_size /= 2
            self.population[i], self.fitness[i] = local_best, local_best_fitness
            if local_best_fitness < self.best_fitness:
                self.best_fitness = local_best_fitness
                self.best_solution = local_best

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
                self.progressive_local_search(func)
                evaluations += self.population_size

        return self.best_solution, self.best_fitness