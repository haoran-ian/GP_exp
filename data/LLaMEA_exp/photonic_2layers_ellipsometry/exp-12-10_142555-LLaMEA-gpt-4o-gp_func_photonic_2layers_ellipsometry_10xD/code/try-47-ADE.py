import numpy as np

class ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # population size
        self.min_CR = 0.1  # minimum crossover probability
        self.max_CR = 0.9  # maximum crossover probability
        self.min_F = 0.4  # minimum mutation factor
        self.max_F = 0.9  # maximum mutation factor
        self.current_evaluations = 0

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def diversity_measure(self, population):
        # Standard deviation of the population to measure diversity
        return np.mean(np.std(population, axis=0))

    def mutate(self, idx, population, F):
        # Select three random indices different from idx
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
            # Adjust mutation factor and crossover rate based on diversity
            diversity = self.diversity_measure(population)
            F = self.min_F + (self.max_F - self.min_F) * (1 - diversity)
            CR = self.min_CR + (self.max_CR - self.min_CR) * diversity

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