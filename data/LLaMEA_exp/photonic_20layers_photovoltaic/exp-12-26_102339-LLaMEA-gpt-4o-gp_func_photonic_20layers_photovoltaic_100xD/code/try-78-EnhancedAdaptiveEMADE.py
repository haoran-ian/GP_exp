import numpy as np

class EnhancedAdaptiveEMADE:
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

        # Strategic mutation rates
        mutation_rates = np.random.uniform(0.6, 1.0, pop_size)
        mutation_scale = np.random.uniform(0.5, 1.0)

        while evals < self.budget:
            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Competitive learning for mutation
                if fitness[i] < np.median(fitness):
                    mutation_rate = np.clip(mutation_rates[i] + np.random.normal(0, 0.05), 0.4, 1.0)
                else:
                    mutation_rate = np.clip(mutation_rates[i] - np.random.normal(0, 0.05), 0.4, 1.0)

                mutant = np.clip(a + mutation_rate * (b - c), lb, ub)

                # Dynamic crossover based on progress and diversity
                progress_ratio = evals / self.budget
                diversity = np.std(pop, axis=0).mean()
                CR = np.clip(0.3 + 0.4 * (1 - diversity / (diversity + 1)) * (1 - progress_ratio), 0.2, 0.9)
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, pop[i])

                # Random Gaussian perturbation
                if np.random.rand() < 0.1:
                    trial += np.random.normal(0, 0.05 * (1 - progress_ratio), self.dim)

                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    mutation_rates[i] = mutation_rate
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial

            if evals / self.budget > 0.2:
                pop_size = max(5, int(self.initial_population_size * (1 - (evals / self.budget)**1.2)))

            if np.random.rand() < 0.1:
                new_individual = np.random.uniform(lb, ub, self.dim)
                new_fitness = func(new_individual)
                evals += 1
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_individual

        return self.best_solution, self.best_fitness