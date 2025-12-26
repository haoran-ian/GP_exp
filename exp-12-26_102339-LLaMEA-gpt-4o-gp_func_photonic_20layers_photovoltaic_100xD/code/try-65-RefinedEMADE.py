import numpy as np

class RefinedEMADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.best_solution = None
        self.best_fitness = np.inf

    def opposition_based_learning(self, individual, lb, ub):
        return lb + ub - individual

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_population_size
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = pop_size
        crossover_rates = np.random.uniform(0.3, 0.9, pop_size)
        F = np.random.uniform(0.5, 0.9, pop_size)

        while evals < self.budget:
            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F[i] * (b - c), lb, ub)

                if np.random.rand() < 0.5:
                    mutant = self.opposition_based_learning(mutant, lb, ub)

                CR = crossover_rates[i]
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    crossover_rates[i] = CR + 0.1 * (1 - CR)  # Reward successful crossover rate
                    F[i] = F[i] + 0.1 * (1 - F[i])
                else:
                    crossover_rates[i] = CR - 0.1 * CR  # Punish unsuccessful crossover rate
                    F[i] = F[i] - 0.1 * F[i]

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

            if evals / self.budget > 0.2:
                pop_size = max(5, int(self.initial_population_size * (1 - (evals / self.budget)**1.5)))

            if np.random.rand() < 0.1:
                new_individual = np.random.uniform(lb, ub, self.dim)
                new_fitness = func(new_individual)
                evals += 1
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_individual

        return self.best_solution, self.best_fitness