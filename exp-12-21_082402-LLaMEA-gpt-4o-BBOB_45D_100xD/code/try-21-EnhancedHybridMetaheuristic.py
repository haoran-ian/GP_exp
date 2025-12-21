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

                # Elitism: Maintain the best solution found so far
                best_idx = np.argmin(fitness)
                best_solution = pop[best_idx]
                best_fitness = fitness[best_idx]

                # Dynamic population size adjustment
                if evals < self.budget * 0.5:
                    pop_size = min(pop_size + 1, self.budget - evals)
                    new_individual = np.random.rand(1, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
                    pop = np.vstack((pop, new_individual))
                    fitness = np.append(fitness, func(new_individual[0]))
                    evals += 1

            return best_solution, best_fitness, evals

        best_solution, best_fitness, evals = differential_evolution()

        def adaptive_neighborhood_search(best_solution, best_fitness, evals):
            neighborhood_size = 0.1 * (bounds[:, 1] - bounds[:, 0])
            while evals < self.budget:
                candidate = best_solution + neighborhood_size * (2 * np.random.rand(self.dim) - 1)
                candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                f_candidate = func(candidate)
                evals += 1
                if f_candidate < best_fitness:
                    best_solution = candidate
                    best_fitness = f_candidate
                if evals >= self.budget:
                    break

        adaptive_neighborhood_search(best_solution, best_fitness, evals)

        return best_solution