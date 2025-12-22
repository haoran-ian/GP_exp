import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop_size = 10
        F_base = 0.8
        CR_base = 0.9

        def stochastic_ranking(pop, fitness, penalty_func):
            return np.argsort(fitness + penalty_func(pop))

        def differential_evolution():
            pop = np.random.rand(pop_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
            fitness = np.array([func(ind) for ind in pop])
            evals = pop_size

            while evals < self.budget:
                for i in range(pop_size):
                    indices = list(range(pop_size))
                    indices.remove(i)
                    a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                    F_dynamic = F_base * (1 - (evals / self.budget))
                    CR_dynamic = CR_base * (1 - ((fitness[i] - min(fitness)) / (max(fitness) - min(fitness) + 1e-9)))
                    mutant = np.clip(a + F_dynamic * (b - c), bounds[:, 0], bounds[:, 1])
                    cross_points = np.random.rand(self.dim) < CR_dynamic
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

            return pop, fitness

        pop, fitness = differential_evolution()
        best_idx = np.argmin(fitness)
        best_ind = pop[best_idx]
        evals = len(fitness)

        def adaptive_neighborhood_search(best_ind, evals):
            initial_neighborhood_size = 0.1 * (bounds[:, 1] - bounds[:, 0])
            neighborhood_size = initial_neighborhood_size

            while evals < self.budget:
                for i in range(pop_size):
                    candidate = best_ind + neighborhood_size * (2 * np.random.rand(self.dim) - 1)
                    candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                    f_candidate = func(candidate)
                    evals += 1
                    if f_candidate < fitness[best_idx]:
                        best_idx = i
                        best_ind = candidate
                        fitness[best_idx] = f_candidate
                        neighborhood_size = initial_neighborhood_size
                    else:
                        neighborhood_size *= 0.9
                    if evals >= self.budget:
                        break

        adaptive_neighborhood_search(best_ind, evals)

        return best_ind