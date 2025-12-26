import numpy as np

class AdaptiveFuzzyEMADE:
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
                self.F = self.fuzzy_F(evals, self.budget)
                mutant = np.clip(a + self.F * (b - c), lb, ub)

                self.CR = self.fuzzy_CR(evals, self.budget)
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

            if np.random.rand() < 0.1:
                new_individual = np.random.uniform(lb, ub, self.dim)
                new_fitness = func(new_individual)
                evals += 1
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_individual

        return self.best_solution, self.best_fitness

    def get_neighbors(self, pop, solution, lb, ub):
        epsilon = 0.1 * (ub - lb)
        neighbors = [np.clip(solution + np.random.uniform(-epsilon, epsilon), lb, ub) for _ in range(5)]
        return neighbors

    def fuzzy_F(self, evals, budget):
        progress = evals / budget
        if progress < 0.3:
            return 0.9
        elif progress < 0.7:
            return 0.5 + 0.4 * np.cos(progress * np.pi)
        else:
            return 0.2

    def fuzzy_CR(self, evals, budget):
        progress = evals / budget
        if progress < 0.3:
            return 0.9
        elif progress < 0.7:
            return 0.5 + 0.5 * np.cos(np.pi * progress)
        else:
            return 0.3