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
        crossover_rates = np.random.uniform(0.3, 0.9, pop_size)
        mutation_scales = np.random.uniform(0.5, 1.0, pop_size)

        while evals < self.budget:
            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Use feedback-adaptive mutation
                if np.random.rand() < 0.5:
                    mutant = np.clip(a + mutation_scales[i] * (b - c), lb, ub)
                else:
                    d = pop[np.random.choice(indices)]
                    mutant = np.clip(a + 0.5 * ((b + c) / 2 - d), lb, ub)

                CR = crossover_rates[i]
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, pop[i])

                # Introduce Gaussian perturbation for diversity
                if np.random.rand() < 0.1:
                    trial += np.random.normal(0, 0.1, self.dim)

                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    crossover_rates[i] = CR + 0.1 * (1 - CR)  # Reward successful crossover rate
                    mutation_scales[i] *= 0.9  # Reduce mutation scale for successful trials
                else:
                    crossover_rates[i] = CR - 0.1 * CR  # Punish unsuccessful crossover rate
                    mutation_scales[i] = min(1.0, mutation_scales[i] * 1.1)  # Increase mutation scale for exploration

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

            # Dynamic population resizing based on progress
            progress = evals / self.budget
            if progress > 0.2:
                pop_size = max(5, int(self.initial_population_size * (1 - progress**1.5)))

            # Introduce new random individual occasionally
            if np.random.rand() < 0.05:
                new_individual = np.random.uniform(lb, ub, self.dim)
                new_fitness = func(new_individual)
                evals += 1
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_individual

        return self.best_solution, self.best_fitness