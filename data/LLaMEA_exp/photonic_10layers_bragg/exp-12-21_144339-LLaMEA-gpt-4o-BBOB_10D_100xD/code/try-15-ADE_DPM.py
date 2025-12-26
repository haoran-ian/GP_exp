import numpy as np

class ADE_DPM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 10 * dim  # Initial Population size
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.eval_count = 0

    def _initialize_population(self, bounds, pop_size):
        return np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))

    def _mutate(self, pop, idx, bounds, population_size_factor):
        indices = list(range(pop.shape[0]))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        adaptive_F = self.F * (1 - self.eval_count / self.budget) * population_size_factor
        mutant = pop[a] + adaptive_F * (pop[b] - pop[c])
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
        pop_size = self.initial_pop_size
        pop = self._initialize_population(bounds, pop_size)
        fitness = np.array([func(ind) for ind in pop])
        self.eval_count = pop_size
        prev_best_fitness = np.min(fitness)

        while self.eval_count < self.budget:
            population_size_factor = (self.budget - self.eval_count) / self.budget
            new_pop = []
            new_fitness = []
            for i in range(len(pop)):
                mutant = self._mutate(pop, i, bounds, population_size_factor)
                trial = self._crossover(pop[i], mutant)
                
                trial_fitness = func(trial)
                self.eval_count += 1
                
                if trial_fitness < fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(trial_fitness)
                else:
                    new_pop.append(pop[i])
                    new_fitness.append(fitness[i])

                current_best_fitness = np.min(new_fitness)
                convergence_rate = (prev_best_fitness - current_best_fitness) / prev_best_fitness if prev_best_fitness != 0 else 0
                prev_best_fitness = current_best_fitness

                if self.eval_count < self.budget:
                    new_pop[-1] = self._adaptive_local_search(new_pop[-1], func, bounds, convergence_rate)
                    new_fitness[-1] = func(new_pop[-1])
                    self.eval_count += 1

                if self.eval_count >= self.budget:
                    break

            pop = np.array(new_pop)
            fitness = np.array(new_fitness)
            # Dynamically adjust population size based on convergence
            pop_size = max(5, int(self.initial_pop_size * population_size_factor))
            if len(pop) > pop_size:
                best_indices = np.argsort(fitness)[:pop_size]
                pop = pop[best_indices]
                fitness = fitness[best_indices]

        best_idx = np.argmin(fitness)
        return pop[best_idx]