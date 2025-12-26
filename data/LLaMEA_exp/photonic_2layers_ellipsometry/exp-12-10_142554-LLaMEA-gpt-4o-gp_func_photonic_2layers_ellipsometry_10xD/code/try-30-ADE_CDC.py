import numpy as np

class ADE_CDC:
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

        def crowding_distance(pop):
            distances = np.zeros(self.population_size)
            for i in range(self.dim):
                sorted_indices = np.argsort(pop[:, i])
                sorted_pop = pop[sorted_indices]
                distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf
                for j in range(1, self.population_size - 1):
                    distances[sorted_indices[j]] += (sorted_pop[j + 1, i] - sorted_pop[j - 1, i])
            return distances

        while eval_count < self.budget:
            distances = crowding_distance(pop)
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)

                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i] or (trial_fitness == fitness[i] and distances[i] > np.median(distances)):
                    pop[i] = trial
                    fitness[i] = trial_fitness

                # Dynamic adaptation
                if eval_count % (self.population_size * 2) == 0:
                    diversity = np.mean([np.linalg.norm(p1 - p2) for p1 in pop for p2 in pop]) / self.dim
                    self.mutation_factor = 0.3 + 0.7 * diversity
                    self.crossover_rate = 0.1 + 0.8 * (1 - diversity)

                if eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]