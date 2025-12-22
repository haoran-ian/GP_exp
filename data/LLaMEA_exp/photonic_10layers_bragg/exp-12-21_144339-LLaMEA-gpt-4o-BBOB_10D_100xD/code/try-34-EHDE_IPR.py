import numpy as np

class EHDE_IPR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.eval_count = 0

    def _initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

    def _mutate(self, pop, idx, bounds):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutation_factor = self.F * (1 - self.eval_count / self.budget)  # Refined mutation precision
        mutant = pop[a] + mutation_factor * (pop[b] - pop[c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def _crossover(self, target, mutant):
        crossover = np.random.rand(self.dim) < self.CR
        return np.where(crossover, mutant, target)

    def _adaptive_local_search(self, candidate, func, bounds, convergence_rate):
        best = candidate
        step_size = 0.01 * (bounds.ub - bounds.lb)
        intensity = int(max(1, self.dim * convergence_rate))
        for _ in range(intensity):
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
        prev_best_fitness = np.min(fitness)

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                mutant = self._mutate(pop, i, bounds)
                trial = self._crossover(pop[i], mutant)
                
                trial_fitness = func(trial)
                self.eval_count += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                current_best_fitness = np.min(fitness)
                convergence_rate = (prev_best_fitness - current_best_fitness) / prev_best_fitness if prev_best_fitness != 0 else 0
                prev_best_fitness = current_best_fitness

                self.CR = 0.5 + 0.4 * convergence_rate  # Dynamically adjust CR based on convergence rate

                if self.eval_count < self.budget:
                    pop[i] = self._adaptive_local_search(pop[i], func, bounds, convergence_rate)
                    fitness[i] = func(pop[i])
                    self.eval_count += 1

                if self.eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx]