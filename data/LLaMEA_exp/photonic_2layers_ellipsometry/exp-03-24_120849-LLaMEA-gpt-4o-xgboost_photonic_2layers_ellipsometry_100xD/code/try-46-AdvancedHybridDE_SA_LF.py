import numpy as np
from scipy.stats import levy

class AdvancedHybridDE_SA_LF:
    def __init__(self, budget, dim, F=0.8, CR=0.9, T0=1000, alpha=0.95, initial_pop_factor=10, max_pop_factor=20):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.T0 = T0
        self.alpha = alpha
        self.initial_pop_factor = initial_pop_factor
        self.max_pop_factor = max_pop_factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        eval_count = 0
        pop_size = self.initial_pop_factor * self.dim
        population = lb + (ub - lb) * np.random.rand(pop_size, self.dim)
        fitness = np.array([func(ind) for ind in population])
        eval_count += pop_size

        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]

        T = self.T0

        while eval_count < self.budget:
            for i in range(pop_size):
                # Adaptive population scaling
                pop_scale = 1 + (self.max_pop_factor - 1) * (eval_count / self.budget)
                adaptive_pop_size = int(pop_scale * self.initial_pop_factor * self.dim)
                if adaptive_pop_size != pop_size:
                    # Resize population if it changes
                    population = np.vstack((
                        population,
                        lb + (ub - lb) * np.random.rand(adaptive_pop_size - pop_size, self.dim)
                    ))[:adaptive_pop_size]
                    fitness = np.array([func(ind) for ind in population])
                    eval_count += adaptive_pop_size - pop_size
                    pop_size = adaptive_pop_size

                # Adaptive Differential Evolution
                indices = np.random.choice(pop_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                F_adaptive = self.F * (1 - eval_count / self.budget) + 0.1
                mutant = np.clip(x0 + F_adaptive * (x1 - x2), lb, ub)
                
                # Dynamic Crossover Rate
                dynamic_CR = self.CR * (1 - eval_count / self.budget) + 0.1
                cross_points = np.random.rand(self.dim) < dynamic_CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial individual
                trial_fitness = func(trial)
                eval_count += 1

                # Selection and Simulated Annealing
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / T):
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness

                # Lévy Flight for exploration
                if np.random.rand() < 0.1:
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

            # Cooling temperature
            T *= self.alpha

        return best, best_fitness