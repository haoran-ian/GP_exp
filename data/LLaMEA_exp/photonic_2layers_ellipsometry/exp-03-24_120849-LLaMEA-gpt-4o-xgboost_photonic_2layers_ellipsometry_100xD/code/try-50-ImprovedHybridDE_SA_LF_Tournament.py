import numpy as np
from scipy.stats import levy

class ImprovedHybridDE_SA_LF_Tournament:
    def __init__(self, budget, dim, F=0.8, CR=0.9, T0=1000, alpha=0.95, population_factor=10, levy_prob=0.15, tournament_size=3):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential evolution parameter
        self.CR = CR  # Crossover probability
        self.T0 = T0  # Initial temperature for Simulated Annealing
        self.alpha = alpha  # Cooling rate
        self.population_factor = population_factor  # Scaling factor for population size
        self.levy_prob = levy_prob  # Probability for Lévy Flight
        self.tournament_size = tournament_size  # Size for tournament selection

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.population_factor * self.dim
        population = lb + (ub - lb) * np.random.rand(pop_size, self.dim)  # Uniform initial sampling
        fitness = np.array([func(ind) for ind in population])
        eval_count = pop_size

        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]

        T = self.T0

        while eval_count < self.budget:
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

                # Selection via tournament
                tournament_indices = np.random.choice(pop_size, self.tournament_size, replace=False)
                tournament_best_idx = tournament_indices[np.argmin(fitness[tournament_indices])]
                if trial_fitness < fitness[tournament_best_idx]:
                    population[tournament_best_idx] = trial
                    fitness[tournament_best_idx] = trial_fitness

                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness

                # Adaptive Lévy Flight for enhanced exploration
                if np.random.rand() < self.levy_prob:  # Adaptive Lévy Flight probability
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

            # Temperature cooling for simulated annealing
            T *= self.alpha

        return best, best_fitness