import numpy as np

class EDADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F_init = 0.5
        self.CR_init = 0.9
        self.eval_count = 0
        self.bounds = None

    def _initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

    def _dynamic_parameters(self):
        progress = self.eval_count / self.budget
        self.F = self.F_init * (1 - progress)
        self.CR = self.CR_init * (1 - progress * 0.5)

    def _mutate(self, pop, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = pop[a] + self.F * (pop[b] - pop[c])
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def _crossover(self, target, mutant):
        crossover = np.random.rand(self.dim) < self.CR
        return np.where(crossover, mutant, target)

    def _hybrid_local_search(self, candidate, func):
        best = candidate
        step_size = 0.01 * (self.bounds.ub - self.bounds.lb)
        intensity = max(1, int(self.dim * 0.1))
        for _ in range(intensity):
            step = np.random.uniform(-step_size, step_size, self.dim)
            neighbor = np.clip(candidate + step, self.bounds.lb, self.bounds.ub)
            if func(neighbor) < func(best):
                best = neighbor
        return best

    def __call__(self, func):
        self.bounds = func.bounds
        pop = self._initialize_population(self.bounds)
        fitness = np.array([func(ind) for ind in pop])
        self.eval_count = self.pop_size

        while self.eval_count < self.budget:
            self._dynamic_parameters()

            for i in range(self.pop_size):
                mutant = self._mutate(pop, i)
                trial = self._crossover(pop[i], mutant)
                
                trial_fitness = func(trial)
                self.eval_count += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                if self.eval_count < self.budget:
                    pop[i] = self._hybrid_local_search(pop[i], func)
                    fitness[i] = func(pop[i])
                    self.eval_count += 1

                if self.eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx]