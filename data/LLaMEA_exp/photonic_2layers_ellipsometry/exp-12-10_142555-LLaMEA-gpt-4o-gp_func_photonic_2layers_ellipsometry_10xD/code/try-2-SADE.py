import numpy as np

class SADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # population size
        self.F = 0.5  # initial mutation factor
        self.CR = 0.9  # initial crossover probability
        self.current_evaluations = 0
        self.successful_F = []
        self.successful_CR = []

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, idx, population):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[indices]
        F = self._adapt_parameter(self.successful_F, 0.1, 0.9)
        return a + F * (b - c)

    def crossover(self, target, donor):
        CR = self._adapt_parameter(self.successful_CR, 0.1, 0.9)
        crossover_mask = np.random.rand(self.dim) < CR
        if not np.any(crossover_mask):  # Guarantee at least one crossover
            crossover_mask[np.random.randint(0, self.dim)] = True
        return np.where(crossover_mask, donor, target)

    def select(self, candidate, target, candidate_fitness, target_fitness):
        if candidate_fitness < target_fitness:
            return candidate, candidate_fitness, True
        return target, target_fitness, False

    def _adapt_parameter(self, successful_params, lower_bound, upper_bound):
        if successful_params:
            return np.mean(successful_params)
        else:
            return np.random.uniform(lower_bound, upper_bound)

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.pop_size

        while self.current_evaluations < self.budget:
            for i in range(self.pop_size):
                donor_vector = self.mutate(i, population)
                trial_vector = self.crossover(population[i], donor_vector)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)

                trial_fitness = func(trial_vector)
                population[i], fitness[i], success = self.select(trial_vector, population[i], trial_fitness, fitness[i])
                self.current_evaluations += 1
                if success:
                    self.successful_F.append(self.F)
                    self.successful_CR.append(self.CR)

                if self.current_evaluations >= self.budget:
                    break

            # Reset successful parameters after some iterations to avoid bias
            if len(self.successful_F) > self.pop_size:
                self.successful_F = []
            if len(self.successful_CR) > self.pop_size:
                self.successful_CR = []

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]