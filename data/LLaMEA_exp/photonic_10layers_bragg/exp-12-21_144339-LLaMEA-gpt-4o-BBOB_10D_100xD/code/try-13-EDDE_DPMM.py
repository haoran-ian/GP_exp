import numpy as np

class EDDE_DPMM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 10 * dim  # Initial population size
        self.pop_size = self.initial_pop_size
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.eval_count = 0

    def _initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

    def _dynamic_population_size(self):
        # Adapt population size based on the budget utilization
        return max(4, int(self.initial_pop_size * (1 - self.eval_count / self.budget)))

    def _multi_phase_mutation(self, pop, idx, bounds):
        # Implement multi-phase mutation strategy
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        
        # Phase 1: Exploration
        mutation_factor1 = self.F * (1 - self.eval_count / self.budget)
        mutant1 = pop[a] + mutation_factor1 * (pop[b] - pop[c])
        
        # Phase 2: Exploitation
        mutation_factor2 = self.F * (self.eval_count / self.budget)
        mutant2 = pop[a] + mutation_factor2 * (pop[b] - pop[c])
        
        # Combine both phases
        mutant = (mutant1 + mutant2) / 2.0
        
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
            self.pop_size = self._dynamic_population_size()
            for i in range(self.pop_size):
                mutant = self._multi_phase_mutation(pop, i, bounds)
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