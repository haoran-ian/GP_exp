import numpy as np

class IDMDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 10 * dim  # initial population size
        self.CR = 0.9  # initial crossover probability
        self.F = 0.5  # mutation factor
        self.current_evaluations = 0
        self.population_size = self.initial_pop_size
        self.min_pop_size = 5

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate(self, idx, population):
        indices = np.random.choice(self.population_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.population_size, 3, replace=False)
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
            return candidate, candidate_fitness
        return target, target_fitness

    def adapt_parameters(self, improvement):
        if improvement < 1e-6 and self.population_size > self.min_pop_size:
            self.population_size = max(self.min_pop_size, int(self.population_size * 0.9))
        elif improvement > 1e-3:
            self.CR = min(1.0, self.CR + 0.05)
        else:
            self.CR = max(0.1, self.CR - 0.05)

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.population_size
        best_fitness = np.min(fitness)

        while self.current_evaluations < self.budget:
            new_population = []
            new_fitness = []

            for i in range(self.population_size):
                donor_vector = self.mutate(i, population)
                trial_vector = self.crossover(population[i], donor_vector)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)

                selected, fit = self.select(trial_vector, population[i], func)
                new_population.append(selected)
                new_fitness.append(fit)
                self.current_evaluations += 1
                if self.current_evaluations >= self.budget:
                    break

            improvement = best_fitness - np.min(new_fitness)
            self.adapt_parameters(improvement)

            population = np.array(new_population)
            fitness = np.array(new_fitness)
            best_fitness = np.min(fitness)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]