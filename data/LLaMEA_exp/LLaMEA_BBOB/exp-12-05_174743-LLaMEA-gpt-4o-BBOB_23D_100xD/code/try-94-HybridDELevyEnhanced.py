import numpy as np

class HybridDELevyEnhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 15 * dim
        self.population_size = self.initial_population_size
        self.mutation_factor = 0.9
        self.crossover_prob = 0.8
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = None
        self.fitness = None
        self.chaotic_sequence = self.generate_chaotic_sequence(self.population_size)

    def levy_flight(self, step_size=0.01):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step_size * step

    def generate_chaotic_sequence(self, length):
        x = np.random.rand()
        sequence = []
        for _ in range(length):
            x = 4.0 * x * (1 - x)
            sequence.append(x)
        return sequence

    def initialize_population(self):
        self.population = np.random.uniform(self.lower_bound, self.upper_bound,
                                            (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])

    def mutation(self, idx, best_idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        dynamic_mutation_factor = self.mutation_factor * (1 - self.budget_used / self.budget)
        diversity = np.std(self.fitness) / np.mean(self.fitness)
        mutation_intensity = 1 + 0.5 * diversity
        mutated = (self.population[a] + 
                   dynamic_mutation_factor * mutation_intensity * (0.5 * (self.population[b] - self.population[c]) +
                                            0.5 * (self.population[best_idx] - self.population[idx])))
        mutated = np.clip(mutated, self.lower_bound, self.upper_bound)
        return mutated

    def crossover(self, target, mutant):
        crossover_vector = np.copy(target)
        rand_idx = np.random.randint(self.dim)
        adaptive_crossover_prob = self.crossover_prob * (1 - self.budget_used / self.budget) + 0.2
        for j in range(self.dim):
            if np.random.rand() < adaptive_crossover_prob or j == rand_idx:
                crossover_vector[j] = mutant[j]
        return crossover_vector

    def select_best(self):
        return np.argmin(self.fitness)

    def chaotic_local_search(self, idx):
        chaotic_factor = self.chaotic_sequence[idx % self.population_size]
        perturbation = chaotic_factor * (self.upper_bound - self.lower_bound) * np.random.randn(self.dim)
        local_solution = np.clip(self.population[idx] + perturbation, self.lower_bound, self.upper_bound)
        return local_solution

    def __call__(self, func):
        self.initialize_population()
        func_evals = 0
        self.budget_used = 0
        while func_evals < self.budget:
            self.evaluate_population(func)
            best_idx = self.select_best()
            for i in range(self.population_size):
                if func_evals >= self.budget:
                    break
                mutant = self.mutation(i, best_idx)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                func_evals += 1
                self.budget_used = func_evals
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                else:
                    local_trial = self.chaotic_local_search(i)
                    local_fitness = func(local_trial)
                    func_evals += 1
                    self.budget_used = func_evals
                    if local_fitness < self.fitness[i]:
                        self.population[i] = local_trial
                        self.fitness[i] = local_fitness
            self.population_size = max(5, int(self.initial_population_size * (1 - self.budget_used / self.budget)))
        return self.population[self.select_best()]