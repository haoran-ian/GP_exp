import numpy as np

class ADE_SAMC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size
        success_mutation = []
        success_crossover = []

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
                    success_mutation.append(self.mutation_factor)
                    success_crossover.append(self.crossover_rate)

                if eval_count >= self.budget:
                    break

            # Self-adaptation of mutation and crossover rates
            if eval_count % (self.population_size * 2) == 0 and success_mutation:
                self.mutation_factor = np.mean(success_mutation)
                self.crossover_rate = np.mean(success_crossover)
                success_mutation.clear()
                success_crossover.clear()

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]