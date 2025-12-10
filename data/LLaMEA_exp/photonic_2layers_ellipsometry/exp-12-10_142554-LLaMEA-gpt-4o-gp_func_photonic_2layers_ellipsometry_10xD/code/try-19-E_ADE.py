import numpy as np

class E_ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.local_search_prob = 0.1  # Probability of performing local search

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size
        
        def local_search(ind):
            perturbation = np.random.normal(scale=(ub - lb) / 100, size=self.dim)
            candidate = np.clip(ind + perturbation, lb, ub)
            candidate_fitness = func(candidate)
            return (candidate, candidate_fitness) if candidate_fitness < func(ind) else (ind, func(ind))

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

                if np.random.rand() < self.local_search_prob:
                    pop[i], fitness[i] = local_search(pop[i])
                    eval_count += 1

                # Dynamic adaptation
                if eval_count % (self.population_size * 2) == 0:
                    fitness_variance = np.var(fitness)
                    self.mutation_factor = 0.4 + 0.6 * np.tanh(fitness_variance)
                    self.crossover_rate = 0.2 + 0.6 * (1 - np.tanh(fitness_variance))

                if eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]