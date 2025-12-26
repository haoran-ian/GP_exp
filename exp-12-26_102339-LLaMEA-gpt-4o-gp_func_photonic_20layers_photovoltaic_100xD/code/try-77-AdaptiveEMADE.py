import numpy as np

class AdaptiveEMADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.best_solution = None
        self.best_fitness = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_population_size
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = pop_size

        # Introduce differential learning rates
        learning_rates = np.random.uniform(0.4, 1.0, pop_size)
        mutation_scale = np.random.uniform(0.5, 1.0)

        while evals < self.budget:
            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                
                if np.random.rand() < 0.5:
                    mutant = np.clip(a + mutation_scale * (b - c), lb, ub)
                else:
                    d = pop[np.random.choice(indices)]
                    mutant = np.clip(a + 0.5 * ((b + c) / 2 - d), lb, ub)

                # Dynamic crossover based on fitness variance and progress ratio
                fitness_variance = np.var(fitness)
                progress_ratio = evals / self.budget
                CR = np.clip(0.5 * (1 - fitness_variance / (fitness_variance + 1)) * (1 - progress_ratio), 0.3, 0.9)
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, pop[i])

                # Gaussian perturbation for diversity
                if np.random.rand() < 0.1:
                    trial += np.random.normal(0, 0.1, self.dim)

                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    learning_rates[i] = learning_rates[i] + 0.1 * (1 - learning_rates[i])  # Reward successful learning rate
                    mutation_scale = mutation_scale * 0.9 + 0.1 * np.abs(fitness[i] - trial_fitness)  # Adapt mutation scale
                else:
                    learning_rates[i] = learning_rates[i] - 0.1 * learning_rates[i]  # Punish unsuccessful learning rate

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

            if evals / self.budget > 0.2:
                pop_size = max(5, int(self.initial_population_size * (1 - (evals / self.budget)**1.5)))

            if np.random.rand() < 0.1:
                new_individual = np.random.uniform(lb, ub, self.dim)
                new_fitness = func(new_individual)
                evals += 1
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_individual

        return self.best_solution, self.best_fitness