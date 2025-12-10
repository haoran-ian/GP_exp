import numpy as np

class ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # population size
        self.CR_initial = 0.9  # initial crossover probability
        self.CR_min = 0.1  # minimum crossover probability
        self.CR_max = 1.0  # maximum crossover probability
        self.F_initial = 0.5  # initial mutation factor
        self.F_min = 0.3  # minimum mutation factor
        self.F_max = 0.8  # maximum mutation factor
        self.current_evaluations = 0

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, idx, population, F):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[indices]
        return a + F * (b - c)

    def crossover(self, target, donor, CR):
        crossover_mask = np.random.rand(self.dim) < CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        return np.where(crossover_mask, donor, target)

    def select(self, candidate, target, func):
        candidate_fitness = func(candidate)
        target_fitness = func(target)
        if candidate_fitness < target_fitness:
            return candidate, candidate_fitness
        return target, target_fitness

    def adapt_parameters(self, fitness):
        diversity = np.std(fitness) / np.mean(fitness)
        CR = self.CR_max - (self.CR_max - self.CR_min) * diversity
        F = self.F_min + (self.F_max - self.F_min) * diversity
        return CR, F

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.pop_size

        while self.current_evaluations < self.budget:
            for i in range(self.pop_size):
                CR, F = self.adapt_parameters(fitness)
                donor_vector = self.mutate(i, population, F)
                trial_vector = self.crossover(population[i], donor_vector, CR)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)
                
                population[i], fitness[i] = self.select(trial_vector, population[i], func)
                self.current_evaluations += 1
                if self.current_evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]