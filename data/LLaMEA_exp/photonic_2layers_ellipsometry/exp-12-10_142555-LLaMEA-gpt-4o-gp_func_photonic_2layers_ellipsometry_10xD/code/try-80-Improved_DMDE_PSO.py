import numpy as np

class Improved_DMDE_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = max(10, 5 * dim)  # dynamic initial population size
        self.CR = 0.9  # crossover probability
        self.F = np.random.rand(self.initial_pop_size)  # dynamic mutation factor
        self.current_evaluations = 0
        self.personal_best = None
        self.personal_best_fitness = np.inf
        self.velocity = np.zeros((self.initial_pop_size, self.dim))  # velocity for PSO component
        self.adaptive_pop_size = self.initial_pop_size

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.adaptive_pop_size, self.dim))

    def mutate(self, idx, population):
        indices = np.random.choice(self.adaptive_pop_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.adaptive_pop_size, 3, replace=False)
        a, b, c = population[indices]
        F_dynamic = 0.5 + 0.5 * np.tanh(self.personal_best_fitness[idx] / np.max(self.personal_best_fitness))  # fitness-based mutation
        return a + F_dynamic * (b - c)

    def crossover(self, target, donor):
        crossover_mask = np.random.rand(self.dim) < self.CR
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

    def update_velocity(self, population):
        w = 0.9 - 0.5 * (self.current_evaluations / self.budget)  # inertia weight decay
        c1 = 1.5 + 0.5 * (self.current_evaluations / self.budget)  # adaptive learning rate
        c2 = 1.5 - 0.5 * (self.current_evaluations / self.budget)  # adaptive learning rate

        for i in range(self.adaptive_pop_size):
            r1, r2 = np.random.rand(2)
            self.velocity[i] = (
                w * self.velocity[i]
                + c1 * r1 * (self.personal_best[i] - population[i])
                + c2 * r2 * (self.global_best - population[i])
            )

    def adaptive_resizing(self, population, fitness):
        # Dynamically resize population based on convergence
        if self.current_evaluations > self.budget / 2:
            self.adaptive_pop_size = max(5, int(self.adaptive_pop_size * 0.9))
            best_indices = np.argsort(fitness)[:self.adaptive_pop_size]
            return population[best_indices], fitness[best_indices]
        return population, fitness

    def opposition_based_learning(self, population, bounds):
        lb, ub = bounds.lb, bounds.ub
        opposition_population = lb + ub - population
        fitness = np.array([func(ind) for ind in opposition_population])
        self.current_evaluations += opposition_population.shape[0]
        return opposition_population, fitness

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.adaptive_pop_size

        self.personal_best = population.copy()
        self.personal_best_fitness = fitness.copy()
        best_idx = np.argmin(fitness)
        self.global_best = population[best_idx]

        while self.current_evaluations < self.budget:
            if self.current_evaluations > self.budget / 2:
                opp_population, opp_fitness = self.opposition_based_learning(population, bounds)
                combined_population = np.vstack((population, opp_population))
                combined_fitness = np.hstack((fitness, opp_fitness))
                best_indices = np.argsort(combined_fitness)[:self.adaptive_pop_size]
                population, fitness = combined_population[best_indices], combined_fitness[best_indices]

            for i in range(self.adaptive_pop_size):
                donor_vector = self.mutate(i, population)
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

            population, fitness = self.adaptive_resizing(population, fitness)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]