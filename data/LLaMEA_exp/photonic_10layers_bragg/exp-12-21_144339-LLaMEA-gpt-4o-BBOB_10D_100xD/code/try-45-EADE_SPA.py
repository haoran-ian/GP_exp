import numpy as np

class EADE_SPA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size
        self.F_min, self.F_max = 0.4, 0.9  # Differential weight range
        self.CR_min, self.CR_max = 0.2, 0.9  # Crossover probability range
        self.eval_count = 0
        self.success_rate = 0.2  # Initial success rate
        self.adaptation_window = 50  # How often to adapt rates

    def _initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

    def _adapt_parameters(self):
        # Adjust mutation factor and crossover probability based on success rate
        self.F = self.F_min + (self.F_max - self.F_min) * self.success_rate
        self.CR = self.CR_min + (self.CR_max - self.CR_min) * self.success_rate

    def _mutate(self, pop, idx, bounds):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = pop[a] + self.F * (pop[b] - pop[c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def _crossover(self, target, mutant):
        crossover = np.random.rand(self.dim) < self.CR
        return np.where(crossover, mutant, target)

    def __call__(self, func):
        bounds = func.bounds
        pop = self._initialize_population(bounds)
        fitness = np.array([func(ind) for ind in pop])
        self.eval_count = self.pop_size
        best_fitness = np.min(fitness)
        success_count = 0

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count % self.adaptation_window == 0:
                    self._adapt_parameters()

                mutant = self._mutate(pop, i, bounds)
                trial = self._crossover(pop[i], mutant)
                
                trial_fitness = func(trial)
                self.eval_count += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    success_count += 1

                if self.eval_count >= self.budget:
                    break
            
            # Update success rate
            self.success_rate = success_count / self.adaptation_window
            success_count = 0

        best_idx = np.argmin(fitness)
        return pop[best_idx]