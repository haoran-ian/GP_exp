import numpy as np

class EnhancedAdaptiveHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.local_search_prob = 0.3
        self.pop = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.bounds = None
        self.evaluations = 0
        self.elite_archive = []
        self.archive_size = 5 * dim
        self.chaotic_sequence = self.generate_multi_scale_chaotic_sequence()

    def generate_multi_scale_chaotic_sequence(self, size=2000):
        x = np.random.rand(size)
        return [(4.0 * x * (1 - x)) + (0.5 * np.sin(2 * np.pi * x)) for x in np.random.rand(size)]

    def initialize_population(self):
        self.pop = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.initial_population_size, self.dim))

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.pop])
        return fitness

    def adaptive_mutation(self, idx, fitness):
        candidates = list(range(len(self.pop)))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        chaos_perturbation = self.chaotic_sequence[self.evaluations % len(self.chaotic_sequence)]
        mutant_vector = np.clip(self.pop[a] + self.mutation_factor * (self.pop[b] - self.pop[c]) + chaos_perturbation,
                                self.bounds.lb, self.bounds.ub)
        return mutant_vector

    def crossover(self, target_vector, mutant_vector):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial_vector = np.where(crossover_mask, mutant_vector, target_vector)
        return trial_vector

    def chaotic_local_search(self, vector, func):
        new_vector = np.copy(vector)
        perturbation_scale = (self.bounds.ub - self.bounds.lb) * 0.05
        for _ in range(self.dim):
            perturbation = np.random.normal(0, 1, self.dim) * perturbation_scale
            candidate_vector = np.clip(new_vector + perturbation, self.bounds.lb, self.bounds.ub)
            if func(candidate_vector) < func(new_vector):
                new_vector = candidate_vector
        return new_vector

    def update_elite_archive(self, vector, fitness):
        if len(self.elite_archive) < self.archive_size:
            self.elite_archive.append((vector, fitness))
        else:
            max_fitness = max(self.elite_archive, key=lambda x: x[1])[1]
            if fitness < max_fitness:
                index_to_replace = next(i for i, v in enumerate(self.elite_archive) if v[1] == max_fitness)
                self.elite_archive[index_to_replace] = (vector, fitness)

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population()
        fitness = self.evaluate_population(func)
        self.best_solution = self.pop[np.argmin(fitness)]
        self.best_fitness = min(fitness)

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

                if trial_fitness < self.best_fitness:
                    self.best_solution = trial_vector
                    self.best_fitness = trial_fitness

                self.update_elite_archive(trial_vector, trial_fitness)

                if self.evaluations >= self.budget:
                    break

            fitness = self.evaluate_population(func)

        return self.best_solution