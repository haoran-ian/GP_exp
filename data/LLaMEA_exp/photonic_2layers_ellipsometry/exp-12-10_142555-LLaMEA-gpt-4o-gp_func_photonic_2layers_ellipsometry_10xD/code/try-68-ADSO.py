import numpy as np

class ADSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.CR = 0.9
        self.F = np.random.rand(self.pop_size)
        self.current_evaluations = 0
        self.personal_best = None
        self.personal_best_fitness = np.inf
        self.velocity = np.zeros((self.pop_size, self.dim))

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def adaptive_mutation(self, idx, population, fitness):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[indices]
        F_adaptive = 1 / (1 + np.exp(-0.1 * (fitness[idx] - self.personal_best_fitness[idx])))
        return a + F_adaptive * (b - c)

    def crossover(self, target, donor):
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

    def update_velocity(self, population):
        w = 0.5
        c1 = 1.5
        c2 = 1.5

        for i in range(self.pop_size):
            r1, r2 = np.random.rand(2)
            self.velocity[i] = (
                w * self.velocity[i]
                + c1 * r1 * (self.personal_best[i] - population[i])
                + c2 * r2 * (self.global_best - population[i])
            )

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.pop_size

        self.personal_best = population.copy()
        self.personal_best_fitness = fitness.copy()
        best_idx = np.argmin(fitness)
        self.global_best = population[best_idx]

        while self.current_evaluations < self.budget:
            for i in range(self.pop_size):
                donor_vector = self.adaptive_mutation(i, population, fitness)
                trial_vector = self.crossover(population[i], donor_vector)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)

                population[i], fitness[i] = self.select(trial_vector, population[i], func)
                if fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best[i] = population[i]
                    self.personal_best_fitness[i] = fitness[i]

                if fitness[i] < self.personal_best_fitness[best_idx]:
                    self.global_best = population[i]
                    best_idx = i

                self.current_evaluations += 1
                if self.current_evaluations >= self.budget:
                    break
            
            self.update_velocity(population)
            population += self.velocity
            population = np.clip(population, bounds.lb, bounds.ub)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]