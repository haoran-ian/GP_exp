import numpy as np

class AdaptiveQuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.quorum_threshold = 0.2
        self.population = None
        self.bounds = None

    def initialize_population(self):
        lb, ub = self.bounds.lb, self.bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.bounds.lb, self.bounds.ub)
        return mutant_vector

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_probability
        trial_vector = np.where(crossover_mask, mutant, target)
        return trial_vector

    def adapt_parameters(self, evaluations):
        progress = evaluations / self.budget
        self.mutation_factor = 0.5 + 0.3 * (1 - progress)
        self.crossover_probability = 0.9 - 0.5 * (1 - progress)

    def quorum_sensing(self, fitness):
        best_idx = np.argmin(fitness)
        best_solution = self.population[best_idx]
        for i in range(self.population_size):
            if fitness[i] > fitness[best_idx] * (1 + self.quorum_threshold):
                self.population[i] = best_solution + np.random.normal(0, 0.1, self.dim)

    def apply_quantum_inspiration(self):
        for i in range(self.population_size):
            r = np.random.rand(self.dim)
            self.population[i] = np.where(r < 0.5, self.population[i], np.random.uniform(self.bounds.lb, self.bounds.ub, self.dim))

    def __call__(self, func):
        self.bounds = func.bounds
        self.population = self.initialize_population()
        fitness = np.array([func(ind) for ind in self.population])
        evaluations = self.population_size

        while evaluations < self.budget:
            self.adapt_parameters(evaluations)
            for i in range(self.population_size):
                mutant_vector = self.mutate(i)
                trial_vector = self.crossover(self.population[i], mutant_vector)
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    self.population[i] = trial_vector
                    fitness[i] = trial_fitness

                if evaluations >= self.budget:
                    break

            self.quorum_sensing(fitness)
            self.apply_quantum_inspiration()

        best_idx = np.argmin(fitness)
        return self.population[best_idx]