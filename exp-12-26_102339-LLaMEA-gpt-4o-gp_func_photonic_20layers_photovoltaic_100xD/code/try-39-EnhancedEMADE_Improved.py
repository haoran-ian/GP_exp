import numpy as np

class EnhancedEMADE_Improved:
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
        pop = self.initialize_population(lb, ub, pop_size)
        fitness = np.array([func(ind) for ind in pop])
        evals = pop_size

        while evals < self.budget:
            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
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

            self.F = 0.5 + 0.4 * np.sin((evals / self.budget)**2 * np.pi)

            if evals < self.budget:
                neighbors = self.get_neighbors(pop, self.best_solution, lb, ub)
                for neighbor in neighbors:
                    ns_fitness = func(neighbor)
                    evals += 1
                    if ns_fitness < self.best_fitness:
                        self.best_fitness = ns_fitness
                        self.best_solution = neighbor

            if evals / self.budget > 0.5:
                pop_size = max(10, int(self.initial_population_size * (1 - (evals / self.budget))))
                pop = self.adapt_population(pop, fitness, pop_size, lb, ub)

        return self.best_solution, self.best_fitness

    def initialize_population(self, lb, ub, pop_size):
        chaos = np.random.rand(pop_size, self.dim)
        pop = lb + (ub - lb) * chaos
        return pop

    def get_neighbors(self, pop, solution, lb, ub):
        epsilon = 0.1 * (ub - lb)
        neighbors = [np.clip(solution + np.random.uniform(-epsilon, epsilon), lb, ub) for _ in range(5)]
        return neighbors

    def adapt_population(self, pop, fitness, new_pop_size, lb, ub):
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices[:new_pop_size]]
        while len(pop) < new_pop_size:
            new_individual = np.random.uniform(lb, ub, self.dim)
            pop = np.vstack([pop, new_individual])
        return pop