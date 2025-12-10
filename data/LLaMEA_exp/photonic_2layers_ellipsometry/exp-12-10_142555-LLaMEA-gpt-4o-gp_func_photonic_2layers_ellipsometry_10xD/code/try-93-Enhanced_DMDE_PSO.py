import numpy as np
from sklearn.cluster import KMeans

class Enhanced_DMDE_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(10, 5 * dim)
        self.current_evaluations = 0
        self.personal_best = None
        self.personal_best_fitness = np.inf
        self.velocity = np.zeros((self.pop_size, self.dim))
        self.selection_pressure = 0.5  # New parameter for self-adaptive selection pressure

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, idx, population):
        selected_indices = np.random.choice(self.pop_size, 3, replace=False)
        while idx in selected_indices:
            selected_indices = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[selected_indices]
        fitness_variance = np.var(self.personal_best_fitness)
        F_dynamic = 0.5 + 0.5 * np.tanh(fitness_variance)
        diversity_factor = np.std(population, axis=0).mean() / self.dim
        return a + F_dynamic * diversity_factor * (b - c)

    def crossover(self, target, donor):
        CR_dynamic = 0.6 + 0.4 * np.sin(self.current_evaluations / self.budget * np.pi)
        crossover_mask = np.random.rand(self.dim) < CR_dynamic
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
        w = 0.9 - 0.5 * (self.current_evaluations / self.budget)
        c1 = 1.5 + 0.5 * (self.current_evaluations / self.budget)
        c2 = 1.5 - 0.5 * (self.current_evaluations / self.budget)

        for i in range(self.pop_size):
            r1, r2 = np.random.rand(2)
            self.velocity[i] = (
                w * self.velocity[i]
                + c1 * r1 * (self.personal_best[i] - population[i])
                + c2 * r2 * (self.global_best - population[i])
            )

    def exploit(self, population, bounds):
        num_clusters = max(2, self.pop_size // 10)
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(population)
        centers = kmeans.cluster_centers_

        for center in centers:
            perturbation = 0.1 * (bounds.ub - bounds.lb) * np.random.normal(size=self.dim)
            candidate = np.clip(center + perturbation, bounds.lb, bounds.ub)
            candidate_fitness = func(candidate)
            if candidate_fitness < np.min(self.personal_best_fitness):
                self.global_best = candidate
                self.global_best_fitness = candidate_fitness

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

            # Trigger exploit phase periodically
            if self.current_evaluations % (self.budget // 10) == 0:
                self.exploit(population, bounds)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]