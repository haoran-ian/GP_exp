import numpy as np

class ADEDL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.initial_CR = 0.9
        self.initial_F = 0.5
        self.current_evaluations = 0
        self.CR = np.full(self.pop_size, self.initial_CR)
        self.F = np.full(self.pop_size, self.initial_F)

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, idx, population):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[indices]
        return a + self.F[idx] * (b - c)

    def crossover(self, target, donor, idx):
        crossover_mask = np.random.rand(self.dim) < self.CR[idx]
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        offspring = np.where(crossover_mask, donor, target)
        return offspring

    def select(self, candidate, target, func, idx):
        candidate_fitness = func(candidate)
        target_fitness = func(target)
        if candidate_fitness < target_fitness:
            self.update_parameters(success=True, idx=idx)
            return candidate, candidate_fitness
        self.update_parameters(success=False, idx=idx)
        return target, target_fitness

    def update_parameters(self, success, idx):
        learning_rate = 0.1
        if success:
            self.CR[idx] = min(1.0, self.CR[idx] + learning_rate * (1.0 - self.CR[idx]))
            self.F[idx] = min(1.0, self.F[idx] + learning_rate * (1.0 - self.F[idx]))
        else:
            self.CR[idx] = max(0.1, self.CR[idx] - learning_rate * (self.CR[idx] - 0.1))
            self.F[idx] = max(0.1, self.F[idx] - learning_rate * (self.F[idx] - 0.1))

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.pop_size

        while self.current_evaluations < self.budget:
            for i in range(self.pop_size):
                donor_vector = self.mutate(i, population)
                trial_vector = self.crossover(population[i], donor_vector, i)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)
                
                population[i], fitness[i] = self.select(trial_vector, population[i], func, i)
                self.current_evaluations += 1
                if self.current_evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]