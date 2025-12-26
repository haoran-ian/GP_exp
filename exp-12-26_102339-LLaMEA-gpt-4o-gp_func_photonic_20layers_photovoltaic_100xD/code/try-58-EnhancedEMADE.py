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

        while evals < self.budget:
            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                
                # Enhanced multi-strategy mutation
                if np.random.rand() < 0.5:
                    mutant = np.clip(a + 0.8 * (b - c), lb, ub)
                elif np.random.rand() < 0.3:
                    mutant = np.clip(a + np.random.uniform(0.5, 1.0) * (b - c), lb, ub)
                else:
                    d = pop[np.random.choice(indices)]
                    mutant = np.clip(a + 0.5 * ((b + c) / 2 - d), lb, ub)

                # Adaptive learning rate
                CR = 0.5 + 0.3 * np.cos(3.14 * (evals / self.budget))
                crossover = np.random.rand(self.dim) < CR
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

            # Fitness-based population resizing
            if evals / self.budget > 0.2:
                avg_fitness = np.mean(fitness)
                pop_size = max(5, int(self.initial_population_size * (1 - (evals / self.budget)**1.2) * (avg_fitness / (avg_fitness + 1))))

            # Random reinitialization for diversification
            if np.random.rand() < 0.1:
                new_individual = np.random.uniform(lb, ub, self.dim)
                new_fitness = func(new_individual)
                evals += 1
                if new_fitness < fitness[np.argmax(fitness)]:
                    replace_idx = np.argmax(fitness)
                    pop[replace_idx] = new_individual
                    fitness[replace_idx] = new_fitness

                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_individual

        return self.best_solution, self.best_fitness