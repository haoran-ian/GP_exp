import numpy as np

class DMDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.CR = 0.9
        self.F = 0.5
        self.current_evaluations = 0

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))
    
    def adaptive_mutation_factor(self, generation):
        """ Adaptive mutation factor based on the current generation. """
        return 0.5 + (0.5 - 0.2) * (1 - generation / self.budget)

    def mutate(self, idx, population, generation):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[indices]
        F = self.adaptive_mutation_factor(generation)
        return a + F * (b - c)
    
    def adaptive_crossover_probability(self, fitness, i):
        """ Adaptive crossover probability based on the fitness of each individual. """
        return self.CR + 0.1 * (fitness[i] / np.max(fitness))

    def crossover(self, target, donor, fitness, i):
        CR = self.adaptive_crossover_probability(fitness, i)
        crossover_mask = np.random.rand(self.dim) < CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        offspring = np.where(crossover_mask, donor, target)
        return offspring

    def select(self, candidate, target, func):
        candidate_fitness = func(candidate)
        target_fitness = func(target)
        if candidate_fitness < target_fitness:
            return candidate, candidate_fitness
        return target, target_fitness

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.pop_size

        while self.current_evaluations < self.budget:
            for i in range(self.pop_size):
                donor_vector = self.mutate(i, population, self.current_evaluations)
                trial_vector = self.crossover(population[i], donor_vector, fitness, i)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)
                
                population[i], fitness[i] = self.select(trial_vector, population[i], func)
                self.current_evaluations += 1
                if self.current_evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]