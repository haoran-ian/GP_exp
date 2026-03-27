import numpy as np

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 15 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.local_search_prob = 0.3
        self.pop = []
        self.best_solution = None
        self.bounds = None
        self.gamma = 0.9
        self.learning_rate = 0.1
        self.perturbation_rate = 0.1

    def initialize_population(self):
        self.pop = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.pop])
        return fitness

    def adaptive_mutation(self, idx, fitness):
        dynamic_factors = self.dynamic_scaling(fitness)
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        chaos_perturbation = self.dynamic_chaos_perturbation(idx, fitness)
        mutant_vector = np.clip(self.pop[a] + dynamic_factors[idx] * (self.pop[b] - self.pop[c]) + chaos_perturbation,
                                self.bounds.lb, self.bounds.ub)
        return mutant_vector

    def dynamic_scaling(self, fitness):
        max_fitness = np.max(fitness)
        min_fitness = np.min(fitness)
        return 0.5 + (self.mutation_factor - 0.5) * (fitness - min_fitness) / (max_fitness - min_fitness + 1e-8)

    def dynamic_chaos_perturbation(self, idx, fitness):
        improvement = (fitness[idx] - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-8)
        self.gamma = max(0.1, self.gamma * (1 - self.learning_rate * improvement))
        # Using Legendre polynomial for chaos-induced perturbation
        perturbation = np.random.uniform(-1, 1, self.dim)
        legendre_chaos = np.polynomial.legendre.legval(perturbation, [0, 1])
        return self.gamma * legendre_chaos

    def crossover(self, target_vector, mutant_vector):
        # Dynamic crossover rate based on current fitness
        self.crossover_rate = 0.9 - 0.5 * np.min(self.evaluate_population)
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial_vector = np.where(crossover_mask, mutant_vector, target_vector)
        return trial_vector

    def localized_adaptive_search(self, vector, func):
        new_vector = np.copy(vector)
        perturbation_scale = self.perturbation_rate * (self.bounds.ub - self.bounds.lb)
        for _ in range(self.dim):
            if np.random.rand() < self.local_search_prob:
                perturbation = np.random.normal(0, 1, self.dim) * perturbation_scale
                candidate_vector = np.clip(new_vector + perturbation, self.bounds.lb, self.bounds.ub)
                if func(candidate_vector) < func(new_vector):
                    new_vector = candidate_vector
        return new_vector

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population()
        fitness = self.evaluate_population(func)
        self.best_solution = self.pop[np.argmin(fitness)]

        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                mutant_vector = self.adaptive_mutation(i, fitness)
                trial_vector = self.crossover(self.pop[i], mutant_vector)
                trial_vector = self.localized_adaptive_search(trial_vector, func)
                trial_fitness = func(trial_vector)

                if trial_fitness < fitness[i]:
                    self.pop[i] = trial_vector
                    fitness[i] = trial_fitness

                if trial_fitness < func(self.best_solution):
                    self.best_solution = trial_vector

        return self.best_solution