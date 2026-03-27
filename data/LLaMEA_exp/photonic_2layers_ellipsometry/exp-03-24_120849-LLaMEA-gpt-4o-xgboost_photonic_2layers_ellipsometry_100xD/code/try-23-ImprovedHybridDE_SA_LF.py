import numpy as np
from scipy.stats import levy

class ImprovedHybridDE_SA_LF:
    def __init__(self, budget, dim, F=0.8, CR=0.9, T0=1000, alpha=0.95, population_factor=10):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.T0 = T0
        self.alpha = alpha
        self.population_factor = population_factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.population_factor * self.dim
        population = lb + (ub - lb) * np.random.beta(0.5, 0.5, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = pop_size
        
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]

        T = self.T0

        while eval_count < self.budget:
            for i in range(pop_size):
                F_adaptive = self.F * (1 - eval_count / self.budget) + 0.1
                indices = np.random.choice(pop_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + F_adaptive * (x1 - x2), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / T):
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness
                
                levy_probability = 0.1 + 0.4 * (eval_count / self.budget)
                if np.random.rand() < levy_probability:
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

                if eval_count % (self.budget // 10) == 0:
                    self.population_factor = np.random.randint(8, 12)
                    pop_size = self.population_factor * self.dim
                    population = population[:pop_size]
                    fitness = fitness[:pop_size]

            T *= self.alpha

        return best, best_fitness