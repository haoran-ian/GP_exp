import numpy as np

class EnhancedHybridDifferentialEvolutionWithQuantum:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 15 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.local_search_prob = 0.3
        self.pop = []
        self.best_solution = None
        self.bounds = None
        self.gamma = 0.9
        self.learning_rate = 0.1
        self.perturbation_rate = 0.1
        self.dynamic_pop_size_factor = 0.5
        self.memory_size = 5
        self.memory = []

    def initialize_population(self):
        self.pop = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.initial_population_size, self.dim))

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.pop])
        return fitness

    def update_memory(self, solution):
        if len(self.memory) < self.memory_size:
            self.memory.append(solution)
        else:
            worst_idx = np.argmax([func(sol) for sol in self.memory])
            if func(solution) < func(self.memory[worst_idx]):
                self.memory[worst_idx] = solution

    def adaptive_mutation(self, idx, fitness):
        dynamic_factors = self.dynamic_scaling(fitness)
        candidates = list(range(len(self.pop)))
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
        return self.gamma * np.random.uniform(-1, 1, self.dim)

    def quantum_inspired_perturbation(self, vector):
        if self.memory:
            memory_vector = self.memory[np.random.randint(0, len(self.memory))]
            quantum_perturbation = np.random.normal(0, 1, self.dim) * (memory_vector - vector)
            return np.clip(vector + quantum_perturbation, self.bounds.lb, self.bounds.ub)
        return vector

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

        evaluations = self.initial_population_size
        while evaluations < self.budget:
            for i in range(len(self.pop)):
                mutant_vector = self.adaptive_mutation(i, fitness)
                trial_vector = self.crossover(self.pop[i], mutant_vector)
                trial_vector = self.chaotic_local_search(trial_vector, func)
                trial_vector = self.quantum_inspired_perturbation(trial_vector)
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    self.pop[i] = trial_vector
                    fitness[i] = trial_fitness

                if trial_fitness < func(self.best_solution):
                    self.best_solution = trial_vector

                self.update_memory(self.best_solution)

                if evaluations >= self.budget:
                    break

            self.dynamically_reduce_population()
            fitness = self.evaluate_population(func)

        return self.best_solution