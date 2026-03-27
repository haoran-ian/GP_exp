import numpy as np

class RefinedEMADE_V2:
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
        elite = pop[np.argmin(fitness)]

        while evals < self.budget:
            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Dynamic mutation strategy
                F = 0.5 + 0.5 * np.random.rand()
                mutant = np.clip(a + F * (b - c), lb, ub)
                
                # Adaptive learning rate
                CR = 0.5 + 0.3 * np.cos(3.14 * (evals / self.budget))
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

            # Adaptive population resizing
            if evals / self.budget > 0.2:
                pop_size = max(5, int(self.initial_population_size * (1 - (evals / self.budget)**1.2)))

            # Elite preservation
            if self.best_fitness < func(elite):
                elite = self.best_solution

            # Random exploration
            if np.random.rand() < 0.1:
                new_individual = np.random.uniform(lb, ub, self.dim)
                new_fitness = func(new_individual)
                evals += 1
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_individual

        return self.best_solution, self.best_fitness