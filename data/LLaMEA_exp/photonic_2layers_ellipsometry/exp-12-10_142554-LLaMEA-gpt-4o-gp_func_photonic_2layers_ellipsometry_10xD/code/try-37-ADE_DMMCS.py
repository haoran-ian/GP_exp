import numpy as np

class ADE_DMMCS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.strategy_switch_frequency = 0.2 * self.budget 

    def mutate(self, pop, i):
        indices = list(range(0, i)) + list(range(i+1, self.population_size))
        a, b, c = pop[np.random.choice(indices, 3, replace=False)]
        return np.clip(a + self.mutation_factor * (b - c), func.bounds.lb, func.bounds.ub)

    def crossover(self, mutant, target):
        crossover = np.random.rand(self.dim) < self.crossover_rate
        return np.where(crossover, mutant, target)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size
        mutation_strategy = self.mutate

        while eval_count < self.budget:
            for i in range(self.population_size):
                mutant = mutation_strategy(pop, i)
                trial = self.crossover(mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                if eval_count % self.strategy_switch_frequency < self.population_size:
                    self.mutation_factor = np.random.rand()
                    self.crossover_rate = np.random.rand()

                if eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]