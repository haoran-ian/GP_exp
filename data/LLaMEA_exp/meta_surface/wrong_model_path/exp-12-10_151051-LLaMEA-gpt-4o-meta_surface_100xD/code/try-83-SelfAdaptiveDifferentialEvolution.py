import numpy as np

class SelfAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        pop_size = 10 * self.dim
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T

        # Initialize population and strategy parameters
        population = np.random.rand(pop_size, self.dim)
        mutation_rates = np.random.uniform(0.4, 0.9, pop_size)
        crossover_rates = np.random.uniform(0.3, 0.9, pop_size)
        
        for i in range(self.dim):
            population[:, i] = bounds[i, 0] + population[:, i] * (bounds[i, 1] - bounds[i, 0])

        # Evaluate initial population
        fitness = np.array([func(ind) for ind in population])
        eval_count = pop_size

        # Optimization loop
        while eval_count < self.budget:
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break

                # Select mutation and crossover rates
                F = mutation_rates[i]
                CR = crossover_rates[i]

                # Mutation strategy
                indices = np.random.choice(pop_size, 5, replace=False)
                x0, x1, x2, x3, x4 = population[indices]
                # Dynamic mutation strategy selection
                if np.random.rand() < 0.7:
                    mutant = np.clip(x0 + F * (x1 - x2) + F * (x3 - x4), bounds[:, 0], bounds[:, 1])
                else:
                    best_index = np.argmin(fitness)
                    x_best = population[best_index]
                    mutant = np.clip(x0 + F * (x_best - x0) + F * (x1 - x2), bounds[:, 0], bounds[:, 1])

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                eval_count += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    mutation_rates[i] = 0.9 * mutation_rates[i] + 0.1 * F  # Self-adaptive mutation rate
                    crossover_rates[i] = 0.9 * crossover_rates[i] + 0.1 * CR  # Self-adaptive crossover rate

                    # Local search reinforcement
                    local_perturb = np.random.normal(0, 0.01 * (1 - eval_count / self.budget), self.dim)
                    perturbed_trial = np.clip(trial + local_perturb, bounds[:, 0], bounds[:, 1])
                    f_perturbed_trial = func(perturbed_trial)
                    eval_count += 1
                    if f_perturbed_trial < f_trial:
                        population[i] = perturbed_trial
                        fitness[i] = f_perturbed_trial

        # Best solution
        best_index = np.argmin(fitness)
        return population[best_index]