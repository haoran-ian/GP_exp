import numpy as np

class HybridChaoticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 15 * dim
        self.base_mutation_factor = 0.8
        self.base_crossover_rate = 0.9
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

    def chaotic_differential_evolution(self, func, chaos_index):
        diversity = np.mean(np.std(self.population, axis=0))
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[indices]
            chaotic_factor = self.chaos_sequence[chaos_index]
            adaptive_mutation_factor = self.base_mutation_factor * (0.5 + chaotic_factor) + diversity * (1 - chaos_index/self.budget)
            adaptive_crossover_rate = self.base_crossover_rate * (0.5 + chaotic_factor) 
            mutant = np.clip(a + adaptive_mutation_factor * (b - c), func.bounds.lb, func.bounds.ub)
            cross_points = np.random.rand(self.dim) < adaptive_crossover_rate
            trial = np.where(cross_points, mutant, self.population[i])
            trial_fitness = func(trial)
            if trial_fitness < self.fitness[i]:
                self.population[i], self.fitness[i] = trial, trial_fitness

    def gradient_based_local_search(self, individual, func, step_size=0.01):
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            perturb = np.zeros(self.dim)
            perturb[i] = step_size
            gradient[i] = (func(individual + perturb) - func(individual - perturb)) / (2 * step_size)
        new_individual = np.clip(individual - step_size * gradient * 0.9 * np.random.rand(), func.bounds.lb, func.bounds.ub)  # Dynamic step size adaptation
        new_fitness = func(new_individual)
        if new_fitness < func(individual):
            return new_individual, new_fitness
        return individual, func(individual)

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
                for i in range(self.population_size):
                    self.population[i], self.fitness[i] = self.gradient_based_local_search(self.population[i], func)
                    evaluations += 1
                    if evaluations >= self.budget:
                        break

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]