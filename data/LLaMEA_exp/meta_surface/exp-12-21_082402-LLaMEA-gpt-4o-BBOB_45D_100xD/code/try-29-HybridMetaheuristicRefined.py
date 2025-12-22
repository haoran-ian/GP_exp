import numpy as np

class HybridMetaheuristicRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop_size = max(4, 10)  # Ensure population is at least 4 for DE
        F = 0.8
        CR = 0.9
        elite_ratio = 0.2

        def differential_evolution():
            pop = np.random.rand(pop_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
            fitness = np.array([func(ind) for ind in pop])
            evals = pop_size

            while evals < self.budget:
                for i in range(pop_size):
                    indices = list(range(pop_size))
                    indices.remove(i)
                    a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                    F_dynamic = F * (1 - (evals / self.budget))
                    mutant = np.clip(a + F_dynamic * (b - c), bounds[:, 0], bounds[:, 1])
                    cross_points = np.random.rand(self.dim) < CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, pop[i])
                    f_trial = func(trial)
                    evals += 1
                    if f_trial < fitness[i]:
                        pop[i] = trial
                        fitness[i] = f_trial
                    if evals >= self.budget:
                        break

            elite_count = max(1, int(elite_ratio * pop_size))
            elite_indices = np.argsort(fitness)[:elite_count]

            return pop, fitness, elite_indices

        pop, fitness, elite_indices = differential_evolution()
        best_idx = elite_indices[0]
        best_ind = pop[best_idx]

        def adaptive_neighborhood_search(best_ind, evals):
            neighborhood_size = 0.1 * (bounds[:, 1] - bounds[:, 0])
            while evals < self.budget:
                for i in elite_indices:
                    candidate = pop[i] + neighborhood_size * (2 * np.random.rand(self.dim) - 1)
                    candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                    f_candidate = func(candidate)
                    evals += 1
                    if f_candidate < fitness[i]:
                        pop[i] = candidate
                        fitness[i] = f_candidate
                        if f_candidate < fitness[best_idx]:
                            best_idx = i
                            best_ind = candidate
                    if evals >= self.budget:
                        break

        adaptive_neighborhood_search(best_ind, len(fitness))

        return best_ind