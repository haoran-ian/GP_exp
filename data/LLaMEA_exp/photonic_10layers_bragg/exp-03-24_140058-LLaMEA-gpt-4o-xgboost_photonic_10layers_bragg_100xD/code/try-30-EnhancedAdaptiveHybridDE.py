import numpy as np

class EnhancedAdaptiveHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20 * dim
        self.mutation_factor = 0.9  
        self.crossover_rate = 0.85
        self.local_search_prob = 0.4  
        self.pop = []
        self.best_solution = None
        self.bounds = None
        self.evaluations = 0
        self.local_search_rate = 0.1
        self.dynamic_pop_size_factor = 0.7
        self.tent_map_r = 1.0

    def initialize_population(self):
        self.pop = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.initial_population_size, self.dim))

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.pop])
        return fitness

    def adaptive_mutation(self, idx, fitness):
        dynamic_factors = self.dynamic_scaling(fitness)
        candidates = list(range(len(self.pop)))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        chaos_perturbation = self.progressive_chaos_perturbation(idx, fitness)
        mutant_vector = np.clip(self.pop[a] + dynamic_factors[idx] * (self.pop[b] - self.pop[c]) + chaos_perturbation,
                                self.bounds.lb, self.bounds.ub)
        return mutant_vector

    def dynamic_scaling(self, fitness):
        max_fitness = np.max(fitness)
        min_fitness = np.min(fitness)
        adjusted_factor = (fitness - min_fitness + 1e-6) / (max_fitness - min_fitness + 1e-8)
        return 0.6 + (self.mutation_factor - 0.6) * adjusted_factor

    def progressive_chaos_perturbation(self, idx, fitness):
        improvement = (fitness[idx] - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-8)
        self.tent_map_r = max(0.1, self.tent_map_r * (1 - self.local_search_rate * improvement))
        chaotic_value = self.tent_map(self.tent_map_r)
        return chaotic_value * np.random.uniform(-1, 1, self.dim)

    def tent_map(self, x):
        return 2 * x if x < 0.5 else 2 * (1 - x)

    def crossover(self, target_vector, mutant_vector):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial_vector = np.where(crossover_mask, mutant_vector, target_vector)
        return trial_vector

    def self_adaptive_local_search(self, vector, func):
        new_vector = np.copy(vector)
        perturbation_scale = self.local_search_rate * (self.bounds.ub - self.bounds.lb)
        for _ in range(self.dim):
            perturbation = np.random.normal(0, 1, self.dim) * perturbation_scale * np.random.choice([-1, 1], self.dim)
            candidate_vector = np.clip(new_vector + perturbation, self.bounds.lb, self.bounds.ub)
            if func(candidate_vector) < func(new_vector):
                new_vector = candidate_vector
        return new_vector

    def dynamically_reduce_population(self):
        reduced_size = int(self.initial_population_size * self.dynamic_pop_size_factor)
        self.pop = self.pop[:reduced_size]

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population()
        fitness = self.evaluate_population(func)
        self.best_solution = self.pop[np.argmin(fitness)]

        while self.evaluations < self.budget:
            for i in range(len(self.pop)):
                mutant_vector = self.adaptive_mutation(i, fitness)
                trial_vector = self.crossover(self.pop[i], mutant_vector)
                if np.random.rand() < self.local_search_prob:
                    trial_vector = self.self_adaptive_local_search(trial_vector, func)
                trial_fitness = func(trial_vector)
                self.evaluations += 1

                if trial_fitness < fitness[i]:
                    self.pop[i] = trial_vector
                    fitness[i] = trial_fitness

                if trial_fitness < func(self.best_solution):
                    self.best_solution = trial_vector

                if self.evaluations >= self.budget:
                    break

            self.dynamically_reduce_population()
            fitness = self.evaluate_population(func)

        return self.best_solution