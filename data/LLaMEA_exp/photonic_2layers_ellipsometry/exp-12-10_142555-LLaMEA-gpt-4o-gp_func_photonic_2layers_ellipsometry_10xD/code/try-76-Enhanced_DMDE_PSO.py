import numpy as np

class Enhanced_DMDE_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(10, 5 * dim)
        self.CR = 0.9
        self.F = np.random.rand(self.pop_size)
        self.current_evaluations = 0
        self.personal_best = None
        self.personal_best_fitness = np.inf
        self.velocity = np.zeros((self.pop_size, self.dim))

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, idx, population):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[indices]
        F_dynamic = 0.5 + 0.5 * np.random.rand()
        return a + F_dynamic * (b - c)

    def crossover(self, target, donor):
        crossover_mask = np.random.rand(self.dim) < self.CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        offspring = np.where(crossover_mask, donor, target)
        return offspring

    def tournament_selection(self, candidates, func):
        selected = np.random.choice(candidates, 2, replace=False)
        fitness1, fitness2 = func(selected[0]), func(selected[1])
        if fitness1 < fitness2:
            return selected[0], fitness1
        else:
            return selected[1], fitness2

    def update_velocity(self, population, iteration, max_iterations):
        w_max, w_min = 0.9, 0.4
        w = w_max - ((w_max - w_min) * iteration / max_iterations)
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

        iterations = 0
        while self.current_evaluations < self.budget:
            for i in range(self.pop_size):
                donor_vector = self.mutate(i, population)
                trial_vector = self.crossover(population[i], donor_vector)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)

                population[i], fitness[i] = self.tournament_selection([trial_vector, population[i]], func)
                if fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best[i] = population[i]
                    self.personal_best_fitness[i] = fitness[i]

                if fitness[i] < self.personal_best_fitness[best_idx]:
                    self.global_best = population[i]
                    best_idx = i

                self.current_evaluations += 1
                if self.current_evaluations >= self.budget:
                    break

            self.update_velocity(population, iterations, self.budget // self.pop_size)
            population += self.velocity
            population = np.clip(population, bounds.lb, bounds.ub)
            iterations += 1

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]