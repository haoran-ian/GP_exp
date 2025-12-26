import numpy as np

class AdvancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop_size = 10
        
        # Self-adaptive parameters
        F_min, F_max = 0.5, 1.0
        CR_min, CR_max = 0.1, 0.9

        def adaptive_differential_evolution():
            pop = np.random.rand(pop_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
            fitness = np.array([func(ind) for ind in pop])
            evals = pop_size

            F = np.random.uniform(F_min, F_max, pop_size)
            CR = np.random.uniform(CR_min, CR_max, pop_size)

            while evals < self.budget:
                for i in range(pop_size):
                    indices = list(range(pop_size))
                    indices.remove(i)
                    a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                    
                    mutant = np.clip(a + F[i] * (b - c), bounds[:, 0], bounds[:, 1])
                    cross_points = np.random.rand(self.dim) < CR[i]
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                        
                    trial = np.where(cross_points, mutant, pop[i])
                    f_trial = func(trial)
                    evals += 1
                    if f_trial < fitness[i]:
                        pop[i], fitness[i] = trial, f_trial
                        F[i] = np.random.uniform(F_min, F_max) # Adaptive mutation factor
                        CR[i] = np.random.uniform(CR_min, CR_max) # Adaptive crossover rate
                    if evals >= self.budget:
                        break

            return pop, fitness

        pop, fitness = adaptive_differential_evolution()
        best_idx = np.argmin(fitness)
        best_ind = pop[best_idx]
        evals = len(fitness)

        def dynamic_neighborhood_reduction(best_ind, evals):
            initial_neighborhood = 0.1 * (bounds[:, 1] - bounds[:, 0])
            neighborhood_size = initial_neighborhood

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
                        neighborhood_size *= 0.9 # Reduce neighborhood size progressively
                    else:
                        neighborhood_size = initial_neighborhood # Reset if not improved
                    if evals >= self.budget:
                        break

        dynamic_neighborhood_reduction(best_ind, evals)

        return best_ind