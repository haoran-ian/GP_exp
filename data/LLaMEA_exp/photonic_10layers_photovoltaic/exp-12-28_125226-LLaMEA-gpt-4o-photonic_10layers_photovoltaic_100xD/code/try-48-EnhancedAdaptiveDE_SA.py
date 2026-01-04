import numpy as np

class EnhancedAdaptiveDE_SA:
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
        self.num_subpopulations = 3
        self.subpopulation_size = self.population_size // self.num_subpopulations

    def opposition_based_learning(self, pop, lb, ub):
        opposite_pop = lb + ub - pop
        return np.clip(opposite_pop, lb, ub)

    def adaptive_cooling_schedule(self):
        self.temperature = self.init_temperature * (1 - self.eval_count / self.budget)

    def adaptive_f(self, subpop_idx):
        scale_factor = (subpop_idx + 1) / self.num_subpopulations  # Different behavior for each subpopulation
        return self.f_min + (self.f_max - self.f_min) * np.exp(-5 * self.eval_count / self.budget) * scale_factor

    def stochastic_ranking(self, population, fitness):
        sort_idx = np.argsort(fitness)
        return population[sort_idx], fitness[sort_idx]

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size

        while self.eval_count < self.budget:
            new_population = []
            for subpop_idx in range(self.num_subpopulations):
                subpop_start = subpop_idx * self.subpopulation_size
                subpop_end = subpop_start + self.subpopulation_size
                subpopulation = population[subpop_start:subpop_end]
                
                for i in range(self.subpopulation_size):
                    idxs = [idx for idx in range(self.subpopulation_size) if idx != i]
                    a, b, c = subpopulation[np.random.choice(idxs, 3, replace=False)]
                    adaptive_f = self.adaptive_f(subpop_idx)
                    mutant = np.clip(a + adaptive_f * (b - c), lb, ub)
                    cross_points = np.random.rand(self.dim) < self.cr
                    trial = np.where(cross_points, mutant, subpopulation[i])

                    trial_fitness = func(trial)
                    self.eval_count += 1

                    if trial_fitness < fitness[subpop_start + i] or np.random.rand() < np.exp(-(trial_fitness - fitness[subpop_start + i]) / self.temperature):
                        new_population.append(trial)
                        fitness[subpop_start + i] = trial_fitness
                    else:
                        new_population.append(subpopulation[i])

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

            population, fitness = self.stochastic_ranking(np.array(new_population), fitness)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]