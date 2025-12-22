import numpy as np

class AMSDE_DEEB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.eval_count = 0
        self.strategies = [self._mutate_rand_1, self._mutate_best_1]
        self.strategy_weights = np.ones(len(self.strategies)) / len(self.strategies)

    def _initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

    def _mutate_rand_1(self, pop, idx, bounds):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = pop[a] + self.F * (pop[b] - pop[c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def _mutate_best_1(self, pop, idx, bounds, best):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b = np.random.choice(indices, 2, replace=False)
        mutant = best + self.F * (pop[a] - pop[b])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def _crossover(self, target, mutant):
        crossover = np.random.rand(self.dim) < self.CR
        return np.where(crossover, mutant, target)

    def _dynamic_exploitation_exploration_balance(self, fitness):
        diversity = np.std(fitness)
        return 1.0 / (1.0 + np.exp(-10 * (diversity - 0.1)))

    def __call__(self, func):
        bounds = func.bounds
        pop = self._initialize_population(bounds)
        fitness = np.array([func(ind) for ind in pop])
        self.eval_count = self.pop_size
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                strategy = np.random.choice(self.strategies, p=self.strategy_weights)
                mutant = strategy(pop, i, bounds, best)
                trial = self._crossover(pop[i], mutant)
                
                trial_fitness = func(trial)
                self.eval_count += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                
                if trial_fitness < fitness[best_idx]:
                    best_idx = i
                    best = trial

                if self.eval_count < self.budget:
                    balance = self._dynamic_exploitation_exploration_balance(fitness)
                    if np.random.rand() < balance:
                        step_size = 0.01 * (bounds.ub - bounds.lb)
                        step = np.random.uniform(-step_size, step_size, self.dim)
                        neighbor = np.clip(pop[i] + step, bounds.lb, bounds.ub)
                        neighbor_fitness = func(neighbor)
                        self.eval_count += 1
                        if neighbor_fitness < fitness[i]:
                            pop[i] = neighbor
                            fitness[i] = neighbor_fitness
                            if neighbor_fitness < fitness[best_idx]:
                                best_idx = i
                                best = neighbor

                if self.eval_count >= self.budget:
                    break

            self.strategy_weights = self.strategy_weights * 0.9 + 0.1 * (fitness == np.min(fitness))

        return pop[best_idx]