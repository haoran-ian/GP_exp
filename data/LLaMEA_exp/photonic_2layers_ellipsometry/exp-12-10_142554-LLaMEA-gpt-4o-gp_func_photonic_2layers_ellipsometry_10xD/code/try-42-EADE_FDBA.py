import numpy as np

class EADE_FDBA:
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
        
        def calculate_diversity(population):
            pairwise_distances = np.array([np.linalg.norm(p1 - p2) for i, p1 in enumerate(population) for p2 in population[i+1:]])
            return np.mean(pairwise_distances) / self.dim

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)

                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                # Multi-objective dynamic adaptation
                if eval_count % (self.population_size * 2) == 0:
                    diversity = calculate_diversity(pop)
                    fitness_improvement = np.mean([np.abs(fitness[i] - func(pop[i])) for i in range(self.population_size)])
                    self.mutation_factor = 0.3 + 0.7 * diversity / (1.0 + fitness_improvement)
                    self.crossover_rate = 0.1 + 0.8 * (1.0 - diversity / (1.0 + fitness_improvement))

                if eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]