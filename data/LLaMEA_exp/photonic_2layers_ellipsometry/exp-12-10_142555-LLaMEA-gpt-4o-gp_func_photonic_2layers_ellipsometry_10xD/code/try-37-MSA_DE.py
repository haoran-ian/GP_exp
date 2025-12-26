import numpy as np

class MSA_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.CR = 0.9
        self.F = 0.5
        self.current_evaluations = 0
        self.success_rate = []

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def adaptive_mutation_factor(self):
        if not self.success_rate:
            return self.F
        success_ratio = np.mean(self.success_rate[-min(10, len(self.success_rate)):])
        adaptive_F = self.F + 0.1 * (success_ratio - 0.5)
        return np.clip(adaptive_F, 0.1, 1.0)

    def mutate(self, idx, population, adaptive_F):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[indices]
        return a + adaptive_F * (b - c)

    def crossover(self, target, donor):
        new_CR = np.random.normal(self.CR, 0.1)
        new_CR = np.clip(new_CR, 0, 1)
        crossover_mask = np.random.rand(self.dim) < new_CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        offspring = np.where(crossover_mask, donor, target)
        return offspring

    def select(self, candidate, target, func):
        candidate_fitness = func(candidate)
        target_fitness = func(target)
        if candidate_fitness < target_fitness:
            self.success_rate.append(1)
            return candidate, candidate_fitness
        self.success_rate.append(0)
        return target, target_fitness

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.pop_size

        while self.current_evaluations < self.budget:
            for i in range(self.pop_size):
                adaptive_F = self.adaptive_mutation_factor()
                donor_vector = self.mutate(i, population, adaptive_F)
                trial_vector = self.crossover(population[i], donor_vector)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)

                population[i], fitness[i] = self.select(trial_vector, population[i], func)
                self.current_evaluations += 1
                if self.current_evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]