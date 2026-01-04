import numpy as np
from scipy.stats import levy

class ImprovedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cr = 0.9
        self.f = 0.8
        self.temperature = 100
        self.cooling_rate = 0.99
        self.eval_count = 0

    def levy_flight(self, step_size):
        return levy.rvs(size=self.dim) * step_size

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
                adaptive_f = self.f * (1 - self.eval_count / self.budget) + 0.1
                mutant = np.clip(a + adaptive_f * (b - c), lb, ub)

                # Dynamic Crossover Rate
                dynamic_cr = self.cr * (1 - self.eval_count / self.budget)

                cross_points = np.random.rand(self.dim) < dynamic_cr
                trial = np.where(cross_points, mutant, population[i])

                # Lévy Flight Exploration
                if np.random.rand() < 0.2:
                    trial += self.levy_flight(0.01 * (ub - lb))

                trial_fitness = func(trial)
                self.eval_count += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp(-(trial_fitness - fitness[i]) / self.temperature):
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])

                if self.eval_count >= self.budget:
                    break

            population = np.array(new_population)
            self.temperature *= self.cooling_rate

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]