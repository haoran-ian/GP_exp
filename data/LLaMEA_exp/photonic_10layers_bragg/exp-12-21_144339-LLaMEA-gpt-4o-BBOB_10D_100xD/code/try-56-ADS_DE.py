import numpy as np

class ADS_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.initial_F = 0.8
        self.initial_CR = 0.9
        self.eval_count = 0

    def _initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

    def _mutate(self, pop, idx, bounds, F):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = pop[a] + F * (pop[b] - pop[c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def _crossover(self, target, mutant, CR):
        crossover = np.random.rand(self.dim) < CR
        return np.where(crossover, mutant, target)

    def _calculate_diversity(self, pop):
        return np.mean(np.std(pop, axis=0))

    def _adaptive_control_parameters(self, diversity):
        F = self.initial_F * (1 + 0.5 * (1 - diversity))
        CR = self.initial_CR * diversity
        return F, CR

    def __call__(self, func):
        bounds = func.bounds
        pop = self._initialize_population(bounds)
        fitness = np.array([func(ind) for ind in pop])
        self.eval_count = self.pop_size

        while self.eval_count < self.budget:
            diversity = self._calculate_diversity(pop)
            F, CR = self._adaptive_control_parameters(diversity)

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                mutant = self._mutate(pop, i, bounds, F)
                trial = self._crossover(pop[i], mutant, CR)
                trial_fitness = func(trial)
                self.eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

        best_idx = np.argmin(fitness)
        return pop[best_idx]