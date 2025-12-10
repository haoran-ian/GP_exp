import numpy as np

class IADE_DPR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = list(range(0, i)) + list(range(i+1, self.population_size))
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)

                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
            
            if eval_count % (self.population_size * 2) == 0:
                diversity = np.std(fitness) / np.mean(fitness)
                self.mutation_factor = 0.3 + 0.7 * diversity
                self.crossover_rate = 0.1 + 0.8 * (1 - diversity)

                if diversity < 0.1 and self.population_size > 4 * self.dim:
                    # Reduce the population if diversity is too low
                    reduction_factor = 0.9
                    self.population_size = int(self.population_size * reduction_factor)
                    pop = pop[:self.population_size]
                    fitness = fitness[:self.population_size]
                elif diversity > 0.2 and self.population_size < self.initial_population_size:
                    # Increase the population if diversity is high
                    increase_factor = 1.1
                    extra_pop_size = int(self.population_size * (increase_factor - 1))
                    extra_pop = np.random.uniform(lb, ub, (extra_pop_size, self.dim))
                    extra_fitness = np.array([func(ind) for ind in extra_pop])
                    eval_count += extra_pop_size
                    pop = np.vstack((pop, extra_pop))
                    fitness = np.concatenate((fitness, extra_fitness))
                    self.population_size = pop.shape[0]

                if eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]