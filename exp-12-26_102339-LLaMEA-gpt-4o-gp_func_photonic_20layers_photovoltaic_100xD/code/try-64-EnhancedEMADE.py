import numpy as np

class EnhancedEMADE:
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
        crossover_rates = np.random.uniform(0.3, 0.9, pop_size)
        mutation_rates = np.random.uniform(0.5, 1.0, pop_size)  # New mutation rates

        while evals < self.budget:
            avg_fitness = np.mean(fitness)  # Added for diversity measurement
            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                
                if np.random.rand() < 0.5:
                    mutant = np.clip(a + mutation_rates[i] * (b - c), lb, ub)  # Adaptive mutation
                else:
                    d = pop[np.random.choice(indices)]
                    mutant = np.clip(a + 0.5 * ((b + c) / 2 - d), lb, ub)

                CR = crossover_rates[i]
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    crossover_rates[i] = CR + 0.1 * (1 - CR)
                    mutation_rates[i] = min(1.0, mutation_rates[i] + 0.1)  # Reward mutation adjustment
                else:
                    crossover_rates[i] = CR - 0.1 * CR
                    mutation_rates[i] = max(0.5, mutation_rates[i] - 0.1)  # Punish mutation adjustment

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

            if np.random.rand() < 0.1:
                new_individual = np.random.uniform(lb, ub, self.dim)
                new_fitness = func(new_individual)
                evals += 1
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_individual

            if np.random.rand() < 0.1 and np.mean(fitness) > avg_fitness:  # Increase diversity
                new_individual = np.random.uniform(lb, ub, self.dim)
                new_fitness = func(new_individual)
                evals += 1
                if new_fitness < np.max(fitness):
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = new_individual
                    fitness[worst_idx] = new_fitness

        return self.best_solution, self.best_fitness