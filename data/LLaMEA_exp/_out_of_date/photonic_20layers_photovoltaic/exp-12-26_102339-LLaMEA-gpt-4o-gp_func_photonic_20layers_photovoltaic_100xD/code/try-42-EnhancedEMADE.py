import numpy as np

class EnhancedEMADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.F = 0.9
        self.CR = 0.7
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
                self.F = 0.5 + 0.4 * np.cos((evals / self.budget) * np.pi)
                mutant = np.clip(a + self.F * (b - c), lb, ub)

                self.CR = 0.5 + 0.5 * np.cos(np.pi * (evals / self.budget))
                crossover = np.random.rand(self.dim) < self.CR
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

            if evals < self.budget:
                neighbors = self.get_neighbors(pop, self.best_solution, lb, ub)
                for neighbor in neighbors:
                    ns_fitness = func(neighbor)
                    evals += 1
                    if ns_fitness < self.best_fitness:
                        self.best_fitness = ns_fitness
                        self.best_solution = neighbor

            if evals / self.budget > 0.3:  # Changed line
                pop_size = max(5, int(self.initial_population_size * (1 - (evals / self.budget)**1.5)))  # Changed line

            if np.random.rand() < 0.1:  # Changed line
                new_individual = np.random.uniform(lb, ub, self.dim)  # Changed line
                new_fitness = func(new_individual)  # Changed line
                evals += 1  # Changed line
                if new_fitness < self.best_fitness:  # Changed line
                    self.best_fitness = new_fitness  # Changed line
                    self.best_solution = new_individual  # Changed line

        return self.best_solution, self.best_fitness

    def get_neighbors(self, pop, solution, lb, ub):
        epsilon = 0.1 * (ub - lb)
        neighbors = [np.clip(solution + np.random.uniform(-epsilon, epsilon), lb, ub) for _ in range(5)]
        return neighbors