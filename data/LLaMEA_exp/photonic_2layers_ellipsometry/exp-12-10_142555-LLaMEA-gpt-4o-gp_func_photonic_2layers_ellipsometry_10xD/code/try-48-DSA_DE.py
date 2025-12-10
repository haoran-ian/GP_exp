import numpy as np

class DSA_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
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

    def adapt_parameters(self, historical_success):
        CR = np.clip(0.9 - 0.5 * np.mean(historical_success[-5:]), 0.1, 0.9)
        F = np.clip(0.5 + 0.3 * np.std(historical_success[-5:]), 0.1, 0.9)
        return CR, F

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.pop_size

        historical_success = []

        while self.current_evaluations < self.budget:
            for i in range(self.pop_size):
                CR, F = self.adapt_parameters(historical_success)
                donor_vector = self.mutate(i, population, F)
                trial_vector = self.crossover(population[i], donor_vector, CR)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)

                new_ind, new_fitness = self.select(trial_vector, population[i], func)
                historical_success.append(new_fitness < fitness[i])
                population[i], fitness[i] = new_ind, new_fitness
                self.current_evaluations += 1
                
                if self.current_evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]