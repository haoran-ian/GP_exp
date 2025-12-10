import numpy as np

class EDE_DL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.historical_data = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size
        self.historical_data = list(zip(pop, fitness))

        while eval_count < self.budget:
            new_pop = []
            new_fitness = []
            for i in range(self.population_size):
                indices = list(range(0, i)) + list(range(i+1, self.population_size))
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, pop[i])

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

            pop = np.array(new_pop)
            fitness = np.array(new_fitness)

            # Update historical data
            self.historical_data.extend(zip(pop, fitness))
            self.historical_data = sorted(self.historical_data, key=lambda x: x[1])[:self.population_size]

            # Dynamic adaptation using historical data
            if eval_count % (self.population_size * 2) == 0:
                diversity = np.std([ind for ind, _ in self.historical_data]) / self.dim
                self.mutation_factor = 0.3 + 0.7 * diversity
                self.crossover_rate = 0.1 + 0.8 * (1 - diversity)

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]