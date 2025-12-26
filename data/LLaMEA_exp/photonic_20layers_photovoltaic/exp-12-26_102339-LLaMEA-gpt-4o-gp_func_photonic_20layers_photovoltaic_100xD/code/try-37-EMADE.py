import numpy as np

class EMADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.F = 0.9  # Changed from 0.8 to 0.9
        self.CR = 0.7
        self.best_solution = None
        self.best_fitness = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_population_size
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = pop_size

        while evals < self.budget:
            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                F_dynamic = self.F * np.random.uniform(0.8, 1.2)  # Introduced random scaling factor
                mutant = np.clip(a + F_dynamic * (b - c), lb, ub)

                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

            self.F = 0.5 + 0.4 * np.sin((evals / self.budget)**2 * np.pi)  # Changed the scaling factor formula
            
            if evals < self.budget:
                local_search_solution = self.adaptive_simplex_search(func, self.best_solution, lb, ub)
                local_search_fitness = func(local_search_solution)
                evals += 1
                if local_search_fitness < self.best_fitness:
                    self.best_fitness = local_search_fitness
                    self.best_solution = local_search_solution

            # Dynamic Population Sizing
            if evals / self.budget > 0.5:
                pop_size = max(10, int(self.initial_population_size * (1 - (evals / self.budget))))

        return self.best_solution, self.best_fitness

    def adaptive_simplex_search(self, func, solution, lb, ub):
        n = len(solution)
        simplex = np.zeros((n + 1, n))
        simplex[0] = solution
        for i in range(1, n + 1):
            y = np.copy(solution)
            y[i - 1] = y[i - 1] + 0.05 * (ub[i - 1] - lb[i - 1])
            simplex[i] = y

        for _ in range(20 + int(10 * np.sin(np.pi * len(solution) / self.dim))):  # Adapt number of iterations
            f_values = np.array([func(x) for x in simplex])
            i_h = np.argmax(f_values)
            i_l = np.argmin(f_values)
            
            centroid = np.mean(simplex[[i for i in range(n + 1) if i != i_h]], axis=0)
            x_r = np.clip(centroid + (centroid - simplex[i_h]), lb, ub)
            f_xr = func(x_r)

            if f_xr < f_values[i_l]:
                x_e = np.clip(centroid + 2 * (centroid - simplex[i_h]), lb, ub)
                f_xe = func(x_e)
                if f_xe < f_xr:
                    simplex[i_h] = x_e
                else:
                    simplex[i_h] = x_r
            elif f_xr < f_values[i_h]:
                simplex[i_h] = x_r
            else:
                for j in range(n + 1):
                    if j != i_l:
                        simplex[j] = simplex[i_l] + 0.5 * (simplex[j] - simplex[i_l])

        return simplex[i_l]