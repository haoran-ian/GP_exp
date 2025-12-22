import numpy as np

class HDE_LS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def _initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

    def _mutate(self, pop, idx):
        # Mutation: DE/rand/1 strategy
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = pop[a] + self.F * (pop[b] - pop[c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def _crossover(self, target, mutant):
        # Binomial crossover
        crossover = np.random.rand(self.dim) < self.CR
        return np.where(crossover, mutant, target)

    def _local_search(self, candidate, func, bounds):
        # Simple gradient-free local search
        best = candidate
        step_size = 0.01 * (bounds.ub - bounds.lb)
        for _ in range(self.dim):
            step = np.random.uniform(-step_size, step_size)
            neighbor = np.clip(candidate + step, bounds.lb, bounds.ub)
            if func(neighbor) < func(best):
                best = neighbor
        return best

    def __call__(self, func):
        bounds = func.bounds
        pop = self._initialize_population(bounds)
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.pop_size

        while eval_count < self.budget:
            for i in range(self.pop_size):
                # Mutation and Crossover
                mutant = self._mutate(pop, i)
                trial = self._crossover(pop[i], mutant)
                
                # Selection
                trial_fitness = func(trial)
                eval_count += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                if eval_count >= self.budget:
                    break

                # Local search on the best individual
                if i == 0:
                    pop[i] = self._local_search(pop[i], func, bounds)
                    fitness[i] = func(pop[i])
                    eval_count += 1

                if eval_count >= self.budget:
                    break

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return pop[best_idx]