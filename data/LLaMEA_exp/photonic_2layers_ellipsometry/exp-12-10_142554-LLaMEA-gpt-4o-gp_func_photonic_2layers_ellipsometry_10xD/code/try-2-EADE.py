import numpy as np

class EADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.shrink_factor = 0.9
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            new_pop = []
            new_fitness = []

            for i in range(self.population_size):
                indices = list(range(0, i)) + list(range(i+1, self.population_size))
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                # Gaussian mutation to improve exploration
                gaussian_mutant = np.clip(mutant + 0.01 * np.random.randn(self.dim), lb, ub)

                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, gaussian_mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(trial_fitness)
                else:
                    new_pop.append(pop[i])
                    new_fitness.append(fitness[i])

                if eval_count >= self.budget:
                    break

            pop, fitness = np.array(new_pop), np.array(new_fitness)
            
            # Population resizing for improved exploitation
            if eval_count % (self.initial_population_size * 5) == 0:
                self.population_size = max(int(self.population_size * self.shrink_factor), 4)

            # Dynamic adaptation of mutation factor and crossover rate
            if eval_count % (self.population_size * 2) == 0:
                diversity = np.std(pop, axis=0).mean() / (ub - lb).mean()
                self.mutation_factor = 0.3 + 0.7 * diversity
                self.crossover_rate = 0.1 + 0.8 * (1 - diversity)

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]