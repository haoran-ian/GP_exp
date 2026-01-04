import numpy as np

class EnhancedHybridDE_SA_Optimized:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cr_min = 0.1
        self.cr_max = 0.9
        self.f_min = 0.4
        self.f_max = 0.9
        self.init_temperature = 100
        self.temperature = self.init_temperature
        self.eval_count = 0
        self.cooling_rate = 0.99
        self.min_population_size = 5 * dim
        self.restart_threshold = 0.1 * dim

    def opposition_based_learning(self, pop, lb, ub):
        opposite_pop = lb + ub - pop
        return np.clip(opposite_pop, lb, ub)

    def adaptive_cooling_schedule(self):
        self.temperature = self.init_temperature * (1 - self.eval_count / self.budget)

    def adaptive_f(self):
        return self.f_min + (self.f_max - self.f_min) * np.exp(-5 * self.eval_count / self.budget)

    def adaptive_cr(self):
        return self.cr_min + (self.cr_max - self.cr_min) * (self.eval_count / self.budget)

    def stochastic_ranking(self, population, fitness):
        sort_idx = np.argsort(fitness)
        return population[sort_idx], fitness[sort_idx]

    def resize_population(self, population, fitness, lb, ub):
        new_size = max(self.min_population_size, int(self.population_size * (1 - self.eval_count / self.budget)))
        if new_size < population.shape[0]:
            indices = np.argsort(fitness)[:new_size]
            return population[indices], fitness[indices]
        else:
            extra_size = new_size - population.shape[0]
            extra_pop = np.random.rand(extra_size, self.dim) * (ub - lb) + lb
            extra_fitness = np.array([func(ind) for ind in extra_pop])
            self.eval_count += extra_size
            return np.vstack((population, extra_pop)), np.hstack((fitness, extra_fitness))

    def restart_population(self, lb, ub):
        return np.random.rand(self.population_size, self.dim) * (ub - lb) + lb

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size

        while self.eval_count < self.budget:
            new_population = []
            for i in range(population.shape[0]):
                idxs = [idx for idx in range(population.shape[0]) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_f = self.adaptive_f()
                mutant = np.clip(a + adaptive_f * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.adaptive_cr()
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                self.eval_count += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp(-(trial_fitness - fitness[i]) / self.temperature):
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])

                if self.eval_count >= self.budget:
                    break

            self.adaptive_cooling_schedule()
            opposite_population = self.opposition_based_learning(new_population, lb, ub)
            opposite_fitness = np.array([func(ind) for ind in opposite_population])
            self.eval_count += len(new_population)

            for j in range(len(new_population)):
                if opposite_fitness[j] < fitness[j]:
                    new_population[j] = opposite_population[j]
                    fitness[j] = opposite_fitness[j]

            population, fitness = self.stochastic_ranking(np.array(new_population), fitness)
            population, fitness = self.resize_population(population, fitness, lb, ub)

            if np.std(fitness) < self.restart_threshold:
                population = self.restart_population(lb, ub)
                fitness = np.array([func(ind) for ind in population])
                self.eval_count += self.population_size

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]