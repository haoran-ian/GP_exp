import numpy as np

class EnhancedEMADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.best_solution = None
        self.best_fitness = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_population_size
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = pop_size

        while evals < self.budget:
            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Adaptive differential mutation with dynamic regrouping
                F = 0.5 + 0.3 * np.sin((2 * np.pi * evals) / self.budget)
                mutant = np.clip(a + F * (b - c), lb, ub)

                # Adaptive crossover
                CR = 0.5 + 0.3 * np.cos(np.pi * (evals / self.budget))
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

            # Dynamic population resizing
            if evals / self.budget > 0.2:
                pop_size = max(5, int(self.initial_population_size * (1 - (evals / self.budget)**1.2)))

            # Periodic local search
            if evals % (self.budget // 10) == 0:
                for j in range(pop_size):
                    local_trial = np.clip(pop[j] + np.random.normal(0, 0.1, self.dim), lb, ub)
                    local_fitness = func(local_trial)
                    evals += 1
                    if local_fitness < fitness[j]:
                        pop[j] = local_trial
                        fitness[j] = local_fitness
                    if local_fitness < self.best_fitness:
                        self.best_fitness = local_fitness
                        self.best_solution = local_trial

        return self.best_solution, self.best_fitness