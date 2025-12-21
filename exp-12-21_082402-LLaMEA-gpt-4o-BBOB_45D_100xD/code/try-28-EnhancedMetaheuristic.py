import numpy as np

class EnhancedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop_size = 10
        F = 0.8
        CR = 0.9

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

            return pop, fitness

        pop, fitness = differential_evolution()
        best_idx = np.argmin(fitness)
        best_ind = pop[best_idx]
        evals = len(fitness)

        def covariance_matrix_adaptive_search(best_ind, evals):
            sigma = 0.3
            C = np.eye(self.dim)
            while evals < self.budget:
                z = np.random.randn(self.dim)
                step = np.dot(C, z)
                candidate = best_ind + sigma * step
                candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                f_candidate = func(candidate)
                evals += 1
                if f_candidate < fitness[best_idx]:
                    best_idx = np.argmin(fitness)
                    best_ind = candidate
                    fitness[best_idx] = f_candidate
                    C = (1 - 0.1) * C + 0.1 * np.outer(z, z)
                if evals >= self.budget:
                    break

        covariance_matrix_adaptive_search(best_ind, evals)

        return best_ind