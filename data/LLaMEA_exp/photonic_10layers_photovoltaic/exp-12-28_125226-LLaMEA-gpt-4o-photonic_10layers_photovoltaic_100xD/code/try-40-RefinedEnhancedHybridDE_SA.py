import numpy as np

class RefinedEnhancedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cr = 0.9
        self.f_min = 0.5
        self.f_max = 0.9
        self.init_temperature = 100
        self.cooling_rate = 0.99
        self.temperature = self.init_temperature
        self.eval_count = 0
        self.elite_archive_size = dim
        self.elite_archive = []

    def opposition_based_learning(self, pop, lb, ub):
        opposite_pop = lb + ub - pop
        return np.clip(opposite_pop, lb, ub)

    def adaptive_cooling_schedule(self):
        self.temperature = self.init_temperature * (1 - self.eval_count / self.budget)

    def adaptive_f(self):
        return self.f_min + (self.f_max - self.f_min) * np.exp(-5 * self.eval_count / self.budget)

    def update_elite_archive(self, candidate, fitness):
        if len(self.elite_archive) < self.elite_archive_size:
            self.elite_archive.append((candidate, fitness))
        else:
            worst_idx = np.argmax([fit for _, fit in self.elite_archive])
            if fitness < self.elite_archive[worst_idx][1]:
                self.elite_archive[worst_idx] = (candidate, fitness)

    def elite_based_mutation(self, a, b, c, lb, ub):
        if self.elite_archive:
            elite = min(self.elite_archive, key=lambda x: x[1])[0]
            mutant = np.clip(a + self.adaptive_f() * (b - c) + 0.5 * (elite - a), lb, ub)
        else:
            mutant = np.clip(a + self.adaptive_f() * (b - c), lb, ub)
        return mutant

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size

        while self.eval_count < self.budget:
            new_population = []
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = self.elite_based_mutation(a, b, c, lb, ub)
                cross_points = np.random.rand(self.dim) < self.cr
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                self.eval_count += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp(-(trial_fitness - fitness[i]) / self.temperature):
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                    self.update_elite_archive(trial, trial_fitness)
                else:
                    new_population.append(population[i])

                if self.eval_count >= self.budget:
                    break

            self.adaptive_cooling_schedule()
            opposite_population = self.opposition_based_learning(new_population, lb, ub)
            opposite_fitness = np.array([func(ind) for ind in opposite_population])
            self.eval_count += self.population_size

            for j in range(self.population_size):
                if opposite_fitness[j] < fitness[j]:
                    new_population[j] = opposite_population[j]
                    fitness[j] = opposite_fitness[j]
                    self.update_elite_archive(opposite_population[j], opposite_fitness[j])

            population = np.array(new_population)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]