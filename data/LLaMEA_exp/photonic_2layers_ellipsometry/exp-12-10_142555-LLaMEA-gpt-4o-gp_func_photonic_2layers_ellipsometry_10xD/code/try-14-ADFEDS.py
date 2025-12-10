import numpy as np

class ADFEDS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.CR = 0.9
        self.F = 0.5
        self.current_evaluations = 0
        self.mutation_change_threshold = 0.1
        self.crossover_change_threshold = 0.05

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, idx, population, fitness):
        sorted_indices = np.argsort(fitness)
        a, b, c = population[sorted_indices[:3]]
        return a + self.F * (b - c)

    def crossover(self, target, donor, fitness):
        sorted_indices = np.argsort(fitness)
        best_idx = sorted_indices[0]
        if fitness[best_idx] < np.mean(fitness):
            self.CR += self.crossover_change_threshold
        else:
            self.CR -= self.crossover_change_threshold
        self.CR = np.clip(self.CR, 0.1, 0.9)
        
        crossover_mask = np.random.rand(self.dim) < self.CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        return np.where(crossover_mask, donor, target)

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
                donor_vector = self.mutate(i, population, fitness)
                trial_vector = self.crossover(population[i], donor_vector, fitness)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)
                
                population[i], fitness[i] = self.select(trial_vector, population[i], func)
                self.current_evaluations += 1
                if self.current_evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]