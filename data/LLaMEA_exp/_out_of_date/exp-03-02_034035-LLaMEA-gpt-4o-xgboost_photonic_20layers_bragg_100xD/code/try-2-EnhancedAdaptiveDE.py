import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = None
        self.fitness = None
        self.memory_bank = []  # Memory bank to store promising solutions
    
    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(low=lb, high=ub, size=(self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
    
    def evaluate_population(self, func):
        for i, individual in enumerate(self.population):
            if self.fitness[i] == np.inf:  # Evaluate only unevaluated individuals
                self.fitness[i] = func(individual)
    
    def differential_evolution(self, func):
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[indices]
            mutant = np.clip(a + self.mutation_factor * (b - c), func.bounds.lb, func.bounds.ub)
            cross_points = np.random.rand(self.dim) < self.crossover_rate
            trial = np.where(cross_points, mutant, self.population[i])
            trial_fitness = func(trial)
            if trial_fitness < self.fitness[i]:
                self.population[i], self.fitness[i] = trial, trial_fitness

    def multi_facet_local_search(self, individual, func, step_size=0.1, max_retries=5):
        best_individual = individual
        best_fitness = func(individual)
        for _ in range(max_retries):
            direction = np.random.uniform(-1, 1, self.dim)
            neighbor = np.clip(best_individual + step_size * direction, func.bounds.lb, func.bounds.ub)
            neighbor_fitness = func(neighbor)
            if neighbor_fitness < best_fitness:
                best_individual, best_fitness = neighbor, neighbor_fitness
                step_size *= 1.2  # Increase step size if improved
            else:
                step_size *= 0.5  # Reduce step size if no improvement
        return best_individual, best_fitness

    def memory_based_local_search(self, func):
        for candidate in self.memory_bank:
            new_individual, new_fitness = self.multi_facet_local_search(candidate, func)
            if new_fitness < min(self.fitness):
                idx = np.argmax(self.fitness)
                self.population[idx], self.fitness[idx] = new_individual, new_fitness
    
    def __call__(self, func):
        self.initialize_population(func.bounds.lb, func.bounds.ub)
        self.evaluate_population(func)
        evaluations = self.population_size

        while evaluations < self.budget:
            self.differential_evolution(func)
            evaluations += self.population_size

            if evaluations < self.budget:
                for i in range(self.population_size):
                    self.population[i], self.fitness[i] = self.multi_facet_local_search(self.population[i], func)
                    if self.fitness[i] < min(self.fitness) * 1.1:  # Store promising solutions
                        self.memory_bank.append(self.population[i])
                evaluations += self.population_size

            if evaluations < self.budget and self.memory_bank:
                self.memory_based_local_search(func)
                evaluations += len(self.memory_bank)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]