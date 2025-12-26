import numpy as np

class EnhancedAdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop_size = 15
        F = 0.8
        CR = 0.9

        def differential_evolution():
            pop = np.random.rand(pop_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
            fitness = np.array([func(ind) for ind in pop])
            evals = pop_size
            F_adapt = F
            pop_adapt_size = pop_size

            while evals < self.budget:
                for i in range(pop_adapt_size):
                    indices = list(range(pop_adapt_size))
                    indices.remove(i)
                    a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                    F_dynamic = F_adapt * (1 - (evals / self.budget))
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
                        F_adapt = max(F_adapt - 0.05, 0.3)  # More gradual F adjustment
                    if evals >= self.budget:
                        break

                # Adaptive population resizing with elitism
                if evals < self.budget and evals % (0.1 * self.budget) == 0:
                    elite_count = max(1, int(0.1 * pop_adapt_size))
                    elite_indices = np.argsort(fitness)[:elite_count]
                    elites = pop[elite_indices]
                    pop_adapt_size = max(5, int(pop_adapt_size * 0.9))
                    pop = np.concatenate((elites, pop[:pop_adapt_size]))
                    fitness = np.array([func(ind) for ind in pop])
                    evals = len(fitness)

            return pop, fitness

        pop, fitness = differential_evolution()
        best_idx = np.argmin(fitness)
        best_ind = pop[best_idx]
        evals = len(fitness)

        def multi_phase_neighborhood_search(best_ind, evals):
            initial_neighborhood_size = 0.1 * (bounds[:, 1] - bounds[:, 0])
            phase_count = 3  # Number of phases with decreasing neighborhood size
            for phase in range(phase_count):
                neighborhood_size = initial_neighborhood_size * (0.5 ** phase)
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

        multi_phase_neighborhood_search(best_ind, evals)

        return best_ind