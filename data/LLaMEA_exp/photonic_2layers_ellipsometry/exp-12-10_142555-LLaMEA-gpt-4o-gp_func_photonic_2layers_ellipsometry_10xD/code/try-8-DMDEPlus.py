import numpy as np

class DMDEPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.CR = 0.9
        self.F_base = 0.5
        self.current_evaluations = 0
        self.elite_size = max(1, self.pop_size // 10)  # Maintain an elite pool

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, idx, population, elite_population, strategy="base"):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.pop_size, 3, replace=False)
        if strategy == "base":
            a, b, c = population[indices]
        elif strategy == "elite":
            elite_indices = np.random.choice(len(elite_population), 2, replace=False)
            a, b = elite_population[elite_indices]
            c = population[indices[0]]
        factor = self.F_base if strategy == "base" else self.F_base * 1.2
        return a + factor * (b - c)

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

    def update_elite_population(self, population, fitness):
        elite_indices = np.argsort(fitness)[:self.elite_size]
        return population[elite_indices]

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.pop_size
        elite_population = self.update_elite_population(population, fitness)

        while self.current_evaluations < self.budget:
            for i in range(self.pop_size):
                strategy = "base" if i % 2 == 0 else "elite"
                donor_vector = self.mutate(i, population, elite_population, strategy)
                trial_vector = self.crossover(population[i], donor_vector)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)
                
                population[i], fitness[i] = self.select(trial_vector, population[i], func)
                self.current_evaluations += 1
                if self.current_evaluations >= self.budget:
                    break

            elite_population = self.update_elite_population(population, fitness)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]