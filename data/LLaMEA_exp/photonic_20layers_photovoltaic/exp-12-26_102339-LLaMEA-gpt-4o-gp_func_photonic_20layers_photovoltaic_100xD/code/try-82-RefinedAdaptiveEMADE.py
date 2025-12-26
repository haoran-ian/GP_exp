import numpy as np

class RefinedAdaptiveEMADE:
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

        # Adaptive parameters
        mutation_scale = 0.8
        crossover_prob = 0.8

        while evals < self.budget:
            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Compute diversity-based scaling factor
                diversity = np.mean(np.linalg.norm(pop - np.mean(pop, axis=0), axis=1))
                dynamic_scale = mutation_scale * (1 + 0.1 * np.random.randn() * diversity)

                mutant = np.clip(a + dynamic_scale * (b - c), lb, ub)

                # Adaptive crossover probability
                progress_ratio = evals / self.budget
                CR = crossover_prob * (1 - progress_ratio)
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, pop[i])

                # Adaptive Gaussian perturbation for exploration
                if np.random.rand() < 0.15:
                    trial += np.random.normal(0, 0.1 * (1 - progress_ratio), self.dim)

                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    mutation_scale *= 0.9 + 0.1 * (1 - trial_fitness / (fitness[i] + 1e-9))
                else:
                    mutation_scale *= 1.2

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

            # Dynamic adjustment of population size
            if evals / self.budget > 0.2:
                pop_size = max(5, int(self.initial_population_size * (1 - np.sqrt(evals / self.budget))))
                pop = pop[:pop_size]
                fitness = fitness[:pop_size]

            # Periodically introduce new random individuals
            if np.random.rand() < 0.1:
                new_individual = np.random.uniform(lb, ub, self.dim)
                new_fitness = func(new_individual)
                evals += 1
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_individual

        return self.best_solution, self.best_fitness