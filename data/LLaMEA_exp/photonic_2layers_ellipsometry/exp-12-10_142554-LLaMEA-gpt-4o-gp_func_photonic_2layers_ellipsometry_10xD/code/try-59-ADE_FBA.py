import numpy as np

class ADE_FBA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.initial_population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.initial_population_size

        while eval_count < self.budget:
            for i in range(len(pop)):
                indices = list(range(0, i)) + list(range(i+1, len(pop)))
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)

                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                if eval_count % (len(pop) * 2) == 0:
                    diversity = np.mean([np.linalg.norm(p1 - p2) for p1 in pop for p2 in pop]) / self.dim
                    self.mutation_factor = 0.3 + 0.7 * diversity
                    self.crossover_rate = 0.1 + 0.8 * (1 - diversity)
                    fitness_var = np.var(fitness)
                    if fitness_var < 0.1:
                        pop = pop[:len(pop)//2]
                        fitness = fitness[:len(pop)//2]
                    
                if eval_count >= self.budget:
                    break
                
        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]