import numpy as np

class EnhancedAdaptiveHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 15 * dim
        self.mutation_factor = 0.85
        self.crossover_rate = 0.9
        self.local_search_prob = 0.35
        self.pop = []
        self.best_solution = None
        self.bounds = None
        self.gamma = 0.9
        self.learning_rate = 0.05
        self.perturbation_rate = 0.1
        self.dynamic_pop_size_factor = 0.5
        self.evaluations = 0
        self.logistic_map_r = 4.0

    def initialize_population(self):
        self.pop = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.initial_population_size, self.dim))

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.pop])
        return fitness

    def adaptive_mutation(self, idx, fitness):
        leader = self.select_leader(fitness)
        candidates = list(range(len(self.pop)))
        candidates.remove(idx)
        candidates.remove(leader)
        a, b = np.random.choice(candidates, 2, replace=False)
        chaos_perturbation = self.progressive_chaos_perturbation(idx, fitness)
        mutant_vector = np.clip(self.pop[idx] + self.mutation_factor * (self.pop[leader] - self.pop[idx]) + 
                                self.mutation_factor * (self.pop[a] - self.pop[b]) + chaos_perturbation,
                                self.bounds.lb, self.bounds.ub)
        return mutant_vector

    def select_leader(self, fitness):
        return np.argmin(fitness)

    def progressive_chaos_perturbation(self, idx, fitness):
        improvement = (fitness[idx] - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-8)
        self.gamma = max(0.1, self.gamma * (1 - self.learning_rate * improvement))
        chaotic_value = self.logistic_map(self.gamma)
        return chaotic_value * np.random.uniform(-1, 1, self.dim)

    def logistic_map(self, x):
        return self.logistic_map_r * x * (1 - x)

    def crossover(self, target_vector, mutant_vector):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial_vector = np.where(crossover_mask, mutant_vector, target_vector)
        return trial_vector

    def chaotic_local_search(self, vector, func):
        new_vector = np.copy(vector)
        perturbation_scale = self.perturbation_rate * (self.bounds.ub - self.bounds.lb)
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
                    trial_vector = self.chaotic_local_search(trial_vector, func)
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