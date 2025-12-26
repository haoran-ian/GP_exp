import numpy as np

class EHDE_IPR_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = max(10, 5 * dim)
        self.final_pop_size = max(10, int(0.5 * dim))
        self.F = 0.5  # Initial differential weight
        self.CR = 0.9  # Initial crossover probability
        self.eval_count = 0

    def _initialize_population(self, bounds, pop_size):
        return np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))

    def _adaptive_mutation(self, pop, idx, bounds, fitness):
        indices = list(range(len(pop)))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        fitness_diversity = np.std(fitness) / (np.mean(fitness) + 1e-8)
        mutation_factor = self.F * (1 + fitness_diversity)
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
        pop_size_range = np.linspace(self.initial_pop_size, self.final_pop_size, num=int(self.budget / self.initial_pop_size))
        pop = self._initialize_population(bounds, int(pop_size_range[0]))
        fitness = np.array([func(ind) for ind in pop])
        self.eval_count = len(pop)
        prev_best_fitness = np.min(fitness)

        while self.eval_count < self.budget:
            current_pop_size = int(pop_size_range[min(self.eval_count // self.initial_pop_size, len(pop_size_range) - 1)])
            for i in range(current_pop_size):
                mutant = self._adaptive_mutation(pop, i, bounds, fitness)
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

            if len(pop) > current_pop_size:
                pop = pop[:current_pop_size]
                fitness = fitness[:current_pop_size]

        best_idx = np.argmin(fitness)
        return pop[best_idx]