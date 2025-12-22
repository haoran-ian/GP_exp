import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_pop_size = 10
        F = 0.8
        CR = 0.9

        def differential_evolution():
            pop_size = initial_pop_size
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
                    CR_dynamic = CR * (0.5 + 0.5 * (evals / self.budget))
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

                if evals >= self.budget:
                    break

                # Dynamic population resizing
                if evals % (self.budget // 4) == 0:
                    new_pop_size = min(pop_size * 2, 50)  # Cap the growth to avoid excessive size
                    if new_pop_size > pop_size:
                        additional_pop = np.random.rand(new_pop_size - pop_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
                        pop = np.vstack((pop, additional_pop))
                        pop_size = new_pop_size
                        fitness = np.append(fitness, [func(ind) for ind in additional_pop])
                        evals += new_pop_size - pop_size

            return pop, fitness

        pop, fitness = differential_evolution()
        best_idx = np.argmin(fitness)
        best_ind = pop[best_idx]
        evals = len(fitness) 

        def adaptive_neighborhood_search(best_ind, evals):
            neighborhood_size = 0.1 * (bounds[:, 1] - bounds[:, 0])
            while evals < self.budget:
                for i in range(initial_pop_size):
                    candidate = best_ind + neighborhood_size * (2 * np.random.rand(self.dim) - 1)
                    candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                    f_candidate = func(candidate)
                    evals += 1
                    if f_candidate < fitness[best_idx]:
                        best_idx = i
                        best_ind = candidate
                        fitness[best_idx] = f_candidate
                    if evals >= self.budget:
                        break

        adaptive_neighborhood_search(best_ind, evals)

        return best_ind