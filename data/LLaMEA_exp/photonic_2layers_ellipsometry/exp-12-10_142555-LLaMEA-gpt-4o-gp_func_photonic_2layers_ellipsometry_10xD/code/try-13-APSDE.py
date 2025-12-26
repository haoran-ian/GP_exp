import numpy as np

class APSDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 10 * dim
        self.pop_size = self.initial_pop_size
        self.CR = 0.9  # crossover probability
        self.F = 0.5  # mutation factor
        self.current_evaluations = 0

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

    def adjust_population_size(self, fitness, stagnation_threshold=50):
        if len(set(fitness[-stagnation_threshold:])) == 1:  # Check for stagnation
            self.pop_size = max(self.initial_pop_size // 2, 4)  # Reduce population
        else:
            self.pop_size = min(self.initial_pop_size, self.pop_size + 1)  # Increase population

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.pop_size

        while self.current_evaluations < self.budget:
            previous_fitness = fitness.copy()

            for i in range(self.pop_size):
                donor_vector = self.mutate(i, population)
                trial_vector = self.crossover(population[i], donor_vector)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)

                population[i], fitness[i] = self.select(trial_vector, population[i], func)
                self.current_evaluations += 1
                if self.current_evaluations >= self.budget:
                    break

            if self.current_evaluations < self.budget:
                self.adjust_population_size(fitness)

            best_idx = np.argmin(fitness)
            population = population[np.argsort(fitness)]
            fitness = np.sort(fitness)
            population = population[:self.pop_size]
            fitness = fitness[:self.pop_size]

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]