import numpy as np

class EADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_population_size
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = pop_size

        while eval_count < self.budget:
            new_pop = []
            new_fitness = []
            
            for i in range(pop_size):
                indices = list(range(0, i)) + list(range(i + 1, pop_size))
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
            best_idx = np.argmin(fitness)
            
            # Dynamic adaptation
            diversity = np.mean([np.linalg.norm(p1 - p2) for p1 in pop for p2 in pop]) / self.dim
            self.mutation_factor = 0.3 + 0.7 * diversity
            self.crossover_rate = 0.1 + 0.8 * (1 - diversity)

            if eval_count % (self.initial_population_size * 2) == 0:
                if diversity < 0.1:
                    pop_size = min(self.initial_population_size * 2, self.budget - eval_count + pop_size)
                    pop = np.vstack((pop, np.random.uniform(lb, ub, (pop_size - len(pop), self.dim))))
                    fitness = np.hstack((fitness, [func(ind) for ind in pop[len(fitness):]]))
                    eval_count += pop_size - len(pop)
                else:
                    pop_size = max(self.initial_population_size // 2, 4)
                    indices = np.argsort(fitness)[:pop_size]
                    pop = pop[indices]
                    fitness = fitness[indices]

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]