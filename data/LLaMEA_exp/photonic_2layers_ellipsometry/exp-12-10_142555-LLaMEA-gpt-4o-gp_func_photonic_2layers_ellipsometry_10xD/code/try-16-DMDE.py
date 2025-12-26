import numpy as np

class DMDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # population size
        self.initial_CR = 0.9  # initial crossover probability
        self.initial_F = 0.5  # initial mutation factor
        self.current_evaluations = 0

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def adaptive_parameters(self, fitness):
        diversity = np.std(fitness)
        CR = self.initial_CR * (1 + diversity / (1 + diversity))
        F = self.initial_F * (1 - diversity / (1 + diversity))
        return CR, F

    def mutate(self, idx, population, F):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[indices]
        return a + F * (b - c)

    def crossover(self, target, donor, CR):
        crossover_mask = np.random.rand(self.dim) < CR
        if not np.any(crossover_mask):  # Guarantee at least one crossover
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
            CR, F = self.adaptive_parameters(fitness)
            for i in range(self.pop_size):
                donor_vector = self.mutate(i, population, F)
                trial_vector = self.crossover(population[i], donor_vector, CR)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)
                
                population[i], fitness[i] = self.select(trial_vector, population[i], func)
                self.current_evaluations += 1
                if self.current_evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]