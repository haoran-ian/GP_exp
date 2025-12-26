import numpy as np

class AdaptiveMultimodalDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.quorum_threshold = 0.2
        self.niche_radius = 0.1
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

    def dynamic_niching(self, population, fitness):
        niche_centers = []
        niche_fitness = []
        for i in range(self.population_size):
            if all(np.linalg.norm(population[i] - np.array(nc)) > self.niche_radius for nc in niche_centers):
                niche_centers.append(population[i])
                niche_fitness.append(fitness[i])
            else:
                j = np.argmin([np.linalg.norm(population[i] - np.array(nc)) for nc in niche_centers])
                if fitness[i] < niche_fitness[j]:
                    niche_centers[j] = population[i]
                    niche_fitness[j] = fitness[i]
        return niche_centers, niche_fitness

    def __call__(self, func):
        self.population = self.initialize_population(func.bounds)
        fitness = np.array([func(ind) for ind in self.population])
        evaluations = self.population_size

        while evaluations < self.budget:
            self.adapt_parameters(evaluations)
            niche_centers, niche_fitness = self.dynamic_niching(self.population, fitness)
            for i in range(len(niche_centers)):
                idxs = [j for j, ind in enumerate(self.population) if np.linalg.norm(ind - np.array(niche_centers[i])) <= self.niche_radius]
                for idx in idxs:
                    mutant_vector = self.mutate(idx, self.population, func.bounds)
                    trial_vector = self.crossover(self.population[idx], mutant_vector)
                    trial_fitness = func(trial_vector)
                    evaluations += 1

                    if trial_fitness < fitness[idx]:
                        self.population[idx] = trial_vector
                        fitness[idx] = trial_fitness

                    if evaluations >= self.budget:
                        break

            self.quorum_sensing(self.population, fitness)

        best_idx = np.argmin(fitness)
        return self.population[best_idx]