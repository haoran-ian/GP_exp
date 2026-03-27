import numpy as np

class ImprovedHybridDE_SA:
    def __init__(self, budget, dim, F=0.8, CR=0.9, T0=1000, alpha=0.95, population_factor=10, min_population_factor=5):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.T0 = T0
        self.alpha = alpha
        self.population_factor = population_factor
        self.min_population_factor = min_population_factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.population_factor * self.dim
        min_pop_size = self.min_population_factor * self.dim
        population = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = pop_size
        
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]

        T = self.T0

        while eval_count < self.budget:
            pop_size = int(min_pop_size + (self.population_factor - self.min_population_factor) * self.dim * (1 - eval_count / self.budget))
            for i in range(pop_size):
                # Adaptive Differential Evolution mutation and crossover
                F_adaptive = self.F * (1 - eval_count / self.budget)
                indices = np.random.choice(pop_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + F_adaptive * (x1 - x2), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial individual
                trial_fitness = func(trial)
                eval_count += 1

                # Selection and Simulated Annealing acceptance
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / T):
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness

            # Temperature cooling for simulated annealing with dynamic adjustment
            T *= self.alpha * (1 - eval_count / self.budget)

        return best, best_fitness