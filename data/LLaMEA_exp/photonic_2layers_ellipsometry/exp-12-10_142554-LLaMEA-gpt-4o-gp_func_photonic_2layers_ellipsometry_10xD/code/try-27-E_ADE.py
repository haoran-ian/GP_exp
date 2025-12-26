import numpy as np

class E_ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.historical_fitness = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size
        self.historical_fitness.append(np.mean(fitness))

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

                # Dynamic adaptation based on historical trends
                if eval_count % (self.population_size * 2) == 0:
                    current_mean_fitness = np.mean(fitness)
                    prev_mean_fitness = self.historical_fitness[-1]
                    improvement = prev_mean_fitness - current_mean_fitness
                    self.historical_fitness.append(current_mean_fitness)

                    diversity = np.mean([np.linalg.norm(p1 - p2) for p1 in pop for p2 in pop]) / self.dim
                    self.mutation_factor = np.clip(self.mutation_factor + 0.1 * improvement / prev_mean_fitness, 0.3, 0.9)
                    self.crossover_rate = np.clip(self.crossover_rate - 0.1 * improvement / prev_mean_fitness, 0.1, 0.9)

                if eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]