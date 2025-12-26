import numpy as np

class EnhancedQuorumDifferentialEvolutionWithLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.quorum_threshold = 0.2
        self.local_search_iterations = 5
        self.population = None

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate(self, idx, population, bounds):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant_vector = np.clip(a + self.mutation_factor * (b - c), bounds.lb, bounds.ub)
        return mutant_vector

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_probability
        trial_vector = np.where(crossover_mask, mutant, target)
        return trial_vector

    def adapt_parameters(self, evaluations):
        progress = evaluations / self.budget
        self.mutation_factor = 0.5 + 0.3 * (1 - progress)
        self.crossover_probability = 0.9 - 0.5 * (1 - progress)

    def quorum_sensing(self, population, fitness):
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        for i in range(self.population_size):
            if fitness[i] > fitness[best_idx] * (1 + self.quorum_threshold):
                population[i] = best_solution + np.random.normal(0, 0.1, self.dim)

    def local_search(self, candidate, func, bounds):
        best_candidate = candidate.copy()
        best_fitness = func(best_candidate)
        for _ in range(self.local_search_iterations):
            perturbation = np.random.normal(0, 0.05, self.dim)
            new_candidate = np.clip(best_candidate + perturbation, bounds.lb, bounds.ub)
            new_fitness = func(new_candidate)
            if new_fitness < best_fitness:
                best_candidate = new_candidate
                best_fitness = new_fitness
        return best_candidate, best_fitness

    def __call__(self, func):
        self.population = self.initialize_population(func.bounds)
        fitness = np.array([func(ind) for ind in self.population])
        evaluations = self.population_size

        while evaluations < self.budget:
            self.adapt_parameters(evaluations)
            for i in range(self.population_size):
                mutant_vector = self.mutate(i, self.population, func.bounds)
                trial_vector = self.crossover(self.population[i], mutant_vector)
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    self.population[i] = trial_vector
                    fitness[i] = trial_fitness

                if evaluations >= self.budget:
                    break

            self.quorum_sensing(self.population, fitness)

            # Apply local search to each individual
            for i in range(self.population_size):
                local_candidate, local_fitness = self.local_search(self.population[i], func, func.bounds)
                if local_fitness < fitness[i]:
                    self.population[i] = local_candidate
                    fitness[i] = local_fitness
                evaluations += self.local_search_iterations
                if evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return self.population[best_idx]