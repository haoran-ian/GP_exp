import numpy as np

class EnhancedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cr = 0.9
        self.f = 0.8
        self.init_temperature = 100
        self.cooling_rate = 0.99
        self.temperature = self.init_temperature
        self.eval_count = 0

    def stochastic_opposition_based_learning(self, pop, lb, ub):
        opposite_pop = lb + ub - pop
        r = np.random.rand(*pop.shape)
        return np.where(r < 0.5, opposite_pop, pop)

    def adaptive_cooling_schedule(self):
        self.temperature = self.init_temperature * (1 - self.eval_count / self.budget)

    def adaptive_scaling_factor(self):
        return self.f * (1 + 0.5 * np.sin(np.pi * self.eval_count / self.budget))

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
                adaptive_f = self.adaptive_scaling_factor()
                mutant = np.clip(a + adaptive_f * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.cr
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
            opposite_population = self.stochastic_opposition_based_learning(np.array(new_population), lb, ub)
            opposite_fitness = np.array([func(ind) for ind in opposite_population])
            self.eval_count += self.population_size

            for j in range(self.population_size):
                if opposite_fitness[j] < fitness[j]:
                    new_population[j] = opposite_population[j]
                    fitness[j] = opposite_fitness[j]

            population = np.array(new_population)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]