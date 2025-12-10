import numpy as np

class ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.CR = 0.9
        self.F = 0.5
        self.current_evaluations = 0
        self.success_history = []

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
        if candidate_fitness < target_fitness:
            self.success_history.append((True, candidate_fitness))
            return candidate, candidate_fitness
        else:
            self.success_history.append((False, target_fitness))
            return target, target_fitness

    def adapt_parameters(self):
        if len(self.success_history) > 10:
            recent_successes = sum(success for success, _ in self.success_history[-10:])
            if recent_successes > 5:
                self.CR = min(1.0, self.CR + 0.1)
                self.F = min(1.0, self.F + 0.1)
            else:
                self.CR = max(0.1, self.CR - 0.1)
                self.F = max(0.1, self.F - 0.1)

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
                
                population[i], fitness[i] = self.select(trial_vector, population[i], func)
                self.current_evaluations += 1
                if self.current_evaluations >= self.budget:
                    break

            self.adapt_parameters()

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]