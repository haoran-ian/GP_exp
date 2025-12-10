import numpy as np

class AdaptiveDMDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.init_pop_size = 10 * dim
        self.CR = 0.9
        self.F_init = 0.5
        self.F = self.F_init
        self.current_evaluations = 0
        self.pop_size = self.init_pop_size

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def dynamic_mutation_factor(self, diversity):
        return self.F_init * np.exp(-diversity)

    def calculate_diversity(self, population):
        center = np.mean(population, axis=0)
        return np.mean(np.linalg.norm(population - center, axis=1))

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
        return np.where(crossover_mask, donor, target)

    def select(self, candidate, target, func):
        candidate_fitness = func(candidate)
        target_fitness = func(target)
        return (candidate, candidate_fitness) if candidate_fitness < target_fitness else (target, target_fitness)

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.pop_size

        while self.current_evaluations < self.budget:
            diversity = self.calculate_diversity(population)
            self.F = self.dynamic_mutation_factor(diversity)

            for i in range(self.pop_size):
                donor_vector = self.mutate(i, population)
                trial_vector = self.crossover(population[i], donor_vector)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)
                
                population[i], fitness[i] = self.select(trial_vector, population[i], func)
                self.current_evaluations += 1

                if self.current_evaluations >= self.budget:
                    break

            # Dynamically adjust the population size based on convergence
            best_idx = np.argmin(fitness)
            if self.current_evaluations % (self.init_pop_size * 2) == 0:
                improvement_rate = np.abs(fitness[best_idx] - np.median(fitness)) / np.mean(fitness)
                if improvement_rate < 0.01:  # If convergence is slow
                    self.pop_size = max(4, self.pop_size // 2)
                    population = population[:self.pop_size]
                    fitness = fitness[:self.pop_size]

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]