import numpy as np

class EnhancedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cr_min = 0.5
        self.cr_max = 0.9
        self.f_min = 0.5
        self.f_max = 0.9
        self.init_temperature = 100
        self.cooling_rate = 0.99
        self.temperature = self.init_temperature
        self.eval_count = 0

    def opposition_based_learning(self, pop, lb, ub):
        opposite_pop = lb + ub - pop
        return np.clip(opposite_pop, lb, ub)

    def adaptive_cooling_schedule(self):
        self.temperature = self.init_temperature * (1 - self.eval_count / self.budget)

    def adaptive_f(self):
        return self.f_min + (self.f_max - self.f_min) * np.exp(-5 * self.eval_count / self.budget)

    def adaptive_cr(self):
        return self.cr_max - (self.cr_max - self.cr_min) * (self.eval_count / self.budget)

    def stochastic_ranking(self, population, fitness):
        sort_idx = np.argsort(fitness)
        return population[sort_idx], fitness[sort_idx]

    def dynamic_subpopulation(self, population, fitness):
        num_subpops = 2
        subpop_size = self.population_size // num_subpops
        indices = np.argsort(fitness)
        subpopulations = []
        for i in range(num_subpops):
            start = i * subpop_size
            end = start + subpop_size
            subpopulations.append(population[indices[start:end]])
        return subpopulations

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size

        while self.eval_count < self.budget:
            subpopulations = self.dynamic_subpopulation(population, fitness)
            new_population = []

            for subpop in subpopulations:
                local_fitness = np.array([func(ind) for ind in subpop])
                for i in range(len(subpop)):
                    idxs = [idx for idx in range(len(subpop)) if idx != i]
                    a, b, c = subpop[np.random.choice(idxs, 3, replace=False)]
                    adaptive_f = self.adaptive_f()
                    mutant = np.clip(a + adaptive_f * (b - c), lb, ub)
                    cr = self.adaptive_cr()
                    cross_points = np.random.rand(self.dim) < cr
                    trial = np.where(cross_points, mutant, subpop[i])

                    trial_fitness = func(trial)
                    self.eval_count += 1

                    if trial_fitness < local_fitness[i] or np.random.rand() < np.exp(-(trial_fitness - local_fitness[i]) / self.temperature):
                        new_population.append(trial)
                        local_fitness[i] = trial_fitness
                    else:
                        new_population.append(subpop[i])

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

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]