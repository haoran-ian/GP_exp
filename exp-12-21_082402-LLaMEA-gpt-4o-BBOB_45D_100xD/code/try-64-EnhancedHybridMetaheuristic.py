import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop_size = 15
        F_base = 0.7
        CR_base = 0.8
        elitism_rate = 0.2

        def differential_evolution():
            pop = np.random.rand(pop_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
            fitness = np.array([func(ind) for ind in pop])
            evals = pop_size
            F_adapt = F_base
            CR_adapt = CR_base

            while evals < self.budget:
                sorted_indices = np.argsort(fitness)
                elite_pop = pop[sorted_indices[:int(elitism_rate * pop_size)]]
                
                for i in range(pop_size):
                    if np.random.rand() < elitism_rate:
                        a, b = np.random.choice(elite_pop, 2, replace=False)
                    else:
                        indices = list(range(pop_size))
                        indices.remove(i)
                        a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                    
                    F_dynamic = F_adapt * (1 - (evals / self.budget))
                    mutant = np.clip(a + F_dynamic * (b - c), bounds[:, 0], bounds[:, 1])
                    cross_points = np.random.rand(self.dim) < CR_adapt
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, pop[i])
                    f_trial = func(trial)
                    evals += 1
                    if f_trial < fitness[i]:
                        pop[i] = trial
                        fitness[i] = f_trial
                        F_adapt = max(F_adapt - 0.05, 0.4)  # Dynamic F adjustment
                        CR_adapt = min(CR_adapt + 0.05, 0.95)  # Dynamic CR adjustment
                    if evals >= self.budget:
                        break

            return pop, fitness

        pop, fitness = differential_evolution()
        best_idx = np.argmin(fitness)
        best_ind = pop[best_idx]
        evals = len(fitness)

        def adaptive_neighborhood_search(best_ind, evals):
            neighborhood_size = 0.05 * (bounds[:, 1] - bounds[:, 0])
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
                    if evals >= self.budget:
                        break

        adaptive_neighborhood_search(best_ind, evals)

        return best_ind