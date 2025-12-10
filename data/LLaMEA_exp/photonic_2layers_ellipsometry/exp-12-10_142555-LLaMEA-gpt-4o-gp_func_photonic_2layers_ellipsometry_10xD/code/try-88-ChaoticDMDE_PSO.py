import numpy as np

class ChaoticDMDE_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(10, 5 * dim)  # dynamic population size
        self.CR = 0.8  # crossover probability
        self.F = np.random.rand(self.pop_size)  # dynamic mutation factor
        self.current_evaluations = 0
        self.personal_best = None
        self.personal_best_fitness = np.inf
        self.velocity = np.zeros((self.pop_size, self.dim))  # velocity for PSO component

    def logistic_map(self, x):
        return 4 * x * (1 - x)

    def chaotic_initialization(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        chaotic_seq = np.random.rand(self.pop_size, self.dim)
        for _ in range(100):  # iterate logistic map
            chaotic_seq = self.logistic_map(chaotic_seq)
        return lb + (ub - lb) * chaotic_seq

    def crowding_distance(self, population, fitness):
        distances = np.zeros(self.pop_size)
        sorted_idx = np.argsort(fitness)
        for i in range(1, self.pop_size - 1):
            distances[sorted_idx[i]] += (fitness[sorted_idx[i + 1]] - fitness[sorted_idx[i - 1]])
        distances[sorted_idx[0]] = distances[sorted_idx[-1]] = np.inf
        return distances

    def select(self, candidate, target, candidate_fitness, target_fitness, crowd_distances, idx):
        if candidate_fitness < target_fitness:
            return candidate, candidate_fitness
        if candidate_fitness == target_fitness:
            if crowd_distances[idx] > crowd_distances[np.where((target == self.personal_best).all(axis=1))[0][0]]:
                return candidate, candidate_fitness
        return target, target_fitness

    def update_velocity(self, population):
        w = 0.9 - 0.5 * (self.current_evaluations / self.budget)  # inertia weight decay
        c1 = 1.5 + 0.5 * (self.current_evaluations / self.budget)  # adaptive learning rate
        c2 = 1.5 - 0.5 * (self.current_evaluations / self.budget)  # adaptive learning rate

        for i in range(self.pop_size):
            r1, r2 = np.random.rand(2)
            self.velocity[i] = (
                w * self.velocity[i]
                + c1 * r1 * (self.personal_best[i] - population[i])
                + c2 * r2 * (self.global_best - population[i])
            )

    def __call__(self, func):
        bounds = func.bounds
        population = self.chaotic_initialization(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.pop_size

        self.personal_best = population.copy()
        self.personal_best_fitness = fitness.copy()
        best_idx = np.argmin(fitness)
        self.global_best = population[best_idx]

        while self.current_evaluations < self.budget:
            crowd_distances = self.crowding_distance(population, fitness)
            for i in range(self.pop_size):
                if self.current_evaluations >= self.budget:
                    break
                donor_vector = self.mutate(i, population)
                trial_vector = self.crossover(population[i], donor_vector)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)

                trial_fitness = func(trial_vector)
                population[i], fitness[i] = self.select(trial_vector, population[i], trial_fitness, fitness[i], crowd_distances, i)
                
                if fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best[i] = population[i]
                    self.personal_best_fitness[i] = fitness[i]

                if fitness[i] < self.personal_best_fitness[best_idx]:
                    self.global_best = population[i]
                    best_idx = i

                self.current_evaluations += 1

            self.update_velocity(population)
            population += self.velocity
            population = np.clip(population, bounds.lb, bounds.ub)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]