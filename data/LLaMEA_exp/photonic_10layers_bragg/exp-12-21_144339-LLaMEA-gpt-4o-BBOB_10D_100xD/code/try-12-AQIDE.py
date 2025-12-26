import numpy as np

class AQIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F_initial = 0.5
        self.CR_initial = 0.9
        self.eval_count = 0

    def _initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

    def _quantum_mutation(self, individual, best, bounds):
        direction = np.random.uniform(-1, 1, self.dim)
        step_size = np.random.uniform(0, np.linalg.norm(bounds.ub - bounds.lb))
        mutant = individual + step_size * direction * np.sign(np.random.rand(self.dim) - 0.5)
        return np.clip(mutant, bounds.lb, bounds.ub)

    def _mutate(self, pop, idx, best, bounds):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        F_dynamic = self.F_initial * (1 - self.eval_count / self.budget)
        mutant = pop[a] + F_dynamic * (best - pop[a]) + F_dynamic * (pop[b] - pop[c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def _crossover(self, target, mutant):
        CR_dynamic = self.CR_initial * (1 - self.eval_count / self.budget)
        crossover = np.random.rand(self.dim) < CR_dynamic
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
            best_idx = np.argmin(fitness)
            best = pop[best_idx]
            for i in range(self.pop_size):
                if np.random.rand() < 0.5:
                    mutant = self._mutate(pop, i, best, bounds)
                else:
                    mutant = self._quantum_mutation(pop[i], best, bounds)

                trial = self._crossover(pop[i], mutant)
                trial_fitness = func(trial)
                self.eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                current_best_fitness = np.min(fitness)
                convergence_rate = (prev_best_fitness - current_best_fitness) / prev_best_fitness if prev_best_fitness != 0 else 0
                prev_best_fitness = current_best_fitness

                if self.eval_count < self.budget:
                    pop[i] = self._adaptive_local_search(pop[i], func, bounds, convergence_rate)
                    fitness[i] = func(pop[i])
                    self.eval_count += 1

                if self.eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx]