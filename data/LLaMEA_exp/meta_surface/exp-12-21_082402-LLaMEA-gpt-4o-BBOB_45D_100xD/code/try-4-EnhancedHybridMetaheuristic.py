import numpy as np

class EnhancedHybridMetaheuristic:
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
            historical_fitness = np.full(self.budget, np.inf)

            while evals < self.budget:
                for i in range(pop_size):
                    indices = list(range(pop_size))
                    indices.remove(i)
                    a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                    F_dynamic = F * (1 - (evals / self.budget)) # Adjusted based on the ratio of current evaluations to budget
                    mutant = np.clip(a + F_dynamic * (b - c), bounds[:, 0], bounds[:, 1])
                    cross_points = np.random.rand(self.dim) < CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, pop[i])
                    f_trial = func(trial)
                    evals += 1
                    historical_fitness[evals-1] = f_trial
                    if f_trial < fitness[i]:
                        pop[i] = trial
                        fitness[i] = f_trial

                    # Dynamic population adjustment
                    if evals >= 10 and evals % 10 == 0:
                        recent_fitness = historical_fitness[evals-10:evals]
                        if np.std(recent_fitness) < 0.01 * np.mean(recent_fitness):
                            pop_size = max(2, pop_size - 1)
                        elif np.std(recent_fitness) > 0.1 * np.mean(recent_fitness):
                            pop_size += 1
                            new_individual = np.random.rand(self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
                            pop = np.vstack((pop, new_individual))
                            fitness = np.append(fitness, func(new_individual))
                            evals += 1

                    if evals >= self.budget:
                        break

            return pop, fitness

        pop, fitness = differential_evolution()
        best_idx = np.argmin(fitness)
        best_ind = pop[best_idx]
        evals = len(fitness)

        def adaptive_neighborhood_search(best_ind, evals):
            neighborhood_size = 0.1 * (bounds[:, 1] - bounds[:, 0])
            while evals < self.budget:
                search_intensity = 1 - (evals / self.budget) # Decrease search intensity over time
                for i in range(pop_size):
                    candidate = best_ind + search_intensity * neighborhood_size * (2 * np.random.rand(self.dim) - 1)
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