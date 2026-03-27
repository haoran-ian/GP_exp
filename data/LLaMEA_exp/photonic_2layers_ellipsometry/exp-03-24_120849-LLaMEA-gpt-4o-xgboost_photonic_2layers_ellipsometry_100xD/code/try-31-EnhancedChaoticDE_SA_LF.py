import numpy as np
from scipy.stats import levy
from numpy.random import default_rng

class EnhancedChaoticDE_SA_LF:
    def __init__(self, budget, dim, F=0.8, CR=0.9, T0=1000, alpha=0.95, population_factor=10):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.T0 = T0
        self.alpha = alpha
        self.population_factor = population_factor

    def logistic_map(self, x):
        return 4 * x * (1 - x)

    def __call__(self, func):
        rng = default_rng()
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.population_factor * self.dim
        population = lb + (ub - lb) * rng.beta(0.5, 0.5, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = pop_size
        
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]

        T = self.T0
        chaotic_sequence = rng.uniform(size=pop_size)

        while eval_count < self.budget:
            for i in range(pop_size):
                chaotic_sequence[i] = self.logistic_map(chaotic_sequence[i])
                F_adaptive = self.F * chaotic_sequence[i] + 0.1

                indices = rng.choice(pop_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + F_adaptive * (x1 - x2), lb, ub)
                cross_points = rng.random(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[rng.integers(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i] or rng.random() < np.exp((fitness[i] - trial_fitness) / T):
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness
                
                if rng.random() < 0.1:
                    levy_step = levy.rvs(size=self.dim)
                    new_position = np.clip(population[i] + levy_step, lb, ub)
                    new_fitness = func(new_position)
                    eval_count += 1

                    if new_fitness < fitness[i]:
                        population[i] = new_position
                        fitness[i] = new_fitness

                        if new_fitness < best_fitness:
                            best = new_position
                            best_fitness = new_fitness

            T *= self.alpha

        return best, best_fitness