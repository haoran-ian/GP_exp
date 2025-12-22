import numpy as np

class EHDE_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size
        self.F_init = 0.8  # Initial differential weight
        self.CR_init = 0.9  # Initial crossover probability
        self.eval_count = 0

    def _initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

    def _adaptive_mutation(self, pop, idx, bounds):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        F_dynamic = self.F_init * np.random.rand() * (1 - self.eval_count / self.budget)
        mutant = pop[a] + F_dynamic * (pop[b] - pop[c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def _adaptive_crossover(self, target, mutant):
        CR_dynamic = self.CR_init * (1 - np.exp(-5 * self.eval_count / self.budget))  # Exponential decay
        crossover = np.random.rand(self.dim) < CR_dynamic
        return np.where(crossover, mutant, target)

    def _stochastic_local_search(self, candidate, func, bounds):
        best = candidate
        step_size = 0.1 * (bounds.ub - bounds.lb) * np.random.rand()
        for _ in range(5):  # Fixed number of local attempts
            step = np.random.uniform(-step_size, step_size, self.dim)
            neighbor = np.clip(candidate + step, bounds.lb, bounds.ub)
            if func(neighbor) < func(best):
                best = neighbor
        return best

    def __call__(self, func):
        bounds = func.bounds
        pop = self._initialize_population(bounds)
        fitness = np.array([func(ind) for ind in pop])
        self.eval_count = self.pop_size

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                mutant = self._adaptive_mutation(pop, i, bounds)
                trial = self._adaptive_crossover(pop[i], mutant)
                
                trial_fitness = func(trial)
                self.eval_count += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                if self.eval_count >= self.budget:
                    break

            for i in range(self.pop_size):
                if self.eval_count < self.budget:
                    local_candidate = self._stochastic_local_search(pop[i], func, bounds)
                    local_fitness = func(local_candidate)
                    self.eval_count += 1
                    if local_fitness < fitness[i]:
                        pop[i] = local_candidate
                        fitness[i] = local_fitness

                if self.eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx]