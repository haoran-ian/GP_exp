import numpy as np
from scipy.stats import levy

class EnhancedHybridDE_SA_LF:
    def __init__(self, budget, dim, F=0.8, CR=0.9, T0=1000, alpha=0.95, initial_population_factor=10):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential evolution parameter
        self.CR = CR  # Crossover probability
        self.T0 = T0  # Initial temperature for Simulated Annealing
        self.alpha = alpha  # Cooling rate
        self.initial_population_factor = initial_population_factor  # Initial scaling factor for population size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_population_factor * self.dim
        population = lb + (ub - lb) * np.random.beta(0.5, 0.5, (pop_size, self.dim))  # Modified initial sampling
        fitness = np.array([func(ind) for ind in population])
        eval_count = pop_size

        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]

        T = self.T0

        while eval_count < self.budget:
            # Dynamic population resizing
            if eval_count / self.budget > 0.5 and pop_size > self.dim:
                pop_size = max(self.dim, int(pop_size * 0.9))
                population = population[:pop_size]
                fitness = fitness[:pop_size]

            for i in range(pop_size):
                # Adaptive Differential Evolution mutation and crossover
                F_adaptive = self.F * (1 - eval_count / self.budget) + 0.1  # Fine-tuned adaptation strategy
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

                # Adaptive Lévy Flight for enhanced exploration
                if np.random.rand() < 0.1:
                    levy_scale = np.power((eval_count / self.budget), 0.5)  # Scaling factor for Lévy flight
                    levy_step = levy.rvs(size=self.dim) * levy_scale
                    new_position = np.clip(population[i] + levy_step, lb, ub)
                    new_fitness = func(new_position)
                    eval_count += 1

                    if new_fitness < fitness[i]:
                        population[i] = new_position
                        fitness[i] = new_fitness

                        if new_fitness < best_fitness:
                            best = new_position
                            best_fitness = new_fitness

            # Temperature cooling for simulated annealing
            T *= self.alpha

        return best, best_fitness