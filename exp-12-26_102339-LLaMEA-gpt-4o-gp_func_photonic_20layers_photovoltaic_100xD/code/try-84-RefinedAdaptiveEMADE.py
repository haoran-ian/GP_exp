import numpy as np

class RefinedAdaptiveEMADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.best_solution = None
        self.best_fitness = np.inf
        self.mutation_scale_as = np.array([0.5] * self.initial_population_size)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_population_size
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = pop_size

        learning_rates = np.random.uniform(0.6, 0.9, pop_size)

        while evals < self.budget:
            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Self-adaptive mutation scale
                self.mutation_scale_as[i] = np.clip(self.mutation_scale_as[i] * (1.0 + np.random.normal(0.0, 0.1)), 0.1, 1.0)
                mutant = np.clip(a + self.mutation_scale_as[i] * (b - c), lb, ub)

                fitness_variance = np.var(fitness)
                progress_ratio = evals / self.budget
                CR = np.clip(0.6 * (1 - fitness_variance / (fitness_variance + 1 + 1e-9)) * (1 - progress_ratio), 0.3, 0.9)
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, pop[i])

                if np.random.rand() < 0.15:
                    trial += np.random.normal(0, 0.1 * (1 - progress_ratio), self.dim)

                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    learning_rates[i] = learning_rates[i] + 0.1 * (1 - learning_rates[i])
                else:
                    learning_rates[i] = learning_rates[i] - 0.1 * learning_rates[i]

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

            if evals / self.budget > 0.2:
                pop_size = max(5, int(self.initial_population_size * (1 - (evals / self.budget) ** 1.3)))

            if np.random.rand() < 0.1:
                new_individual = np.random.uniform(lb, ub, self.dim)
                new_fitness = func(new_individual)
                evals += 1
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_individual

        return self.best_solution, self.best_fitness