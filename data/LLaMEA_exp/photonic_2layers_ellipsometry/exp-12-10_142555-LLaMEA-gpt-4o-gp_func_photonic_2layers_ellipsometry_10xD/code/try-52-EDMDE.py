import numpy as np

class EDMDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # population size
        self.CR = 0.9  # crossover probability
        self.F = 0.5  # mutation factor
        self.current_evaluations = 0

    def adapt_parameters(self):
        self.F = 0.5 + np.random.rand() * 0.3  # Adapt F randomly within a range
        self.CR = 0.8 + np.random.rand() * 0.2  # Adapt CR randomly within a range

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, idx, population):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[indices]
        return a + self.F * (b - c)

    def crossover(self, target, donor):
        crossover_mask = np.random.rand(self.dim) < self.CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        offspring = np.where(crossover_mask, donor, target)
        return offspring

    def select(self, candidate, target, func):
        candidate_fitness = func(candidate)
        target_fitness = func(target)
        prob = 1 / (1 + np.exp(-(candidate_fitness - target_fitness)))
        if np.random.rand() < prob:
            return candidate, candidate_fitness
        return target, target_fitness

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.pop_size

        while self.current_evaluations < self.budget:
            self.adapt_parameters()
            for i in range(self.pop_size):
                donor_vector = self.mutate(i, population)
                trial_vector = self.crossover(population[i], donor_vector)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)
                
                population[i], fitness[i] = self.select(trial_vector, population[i], func)
                self.current_evaluations += 1
                if self.current_evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]