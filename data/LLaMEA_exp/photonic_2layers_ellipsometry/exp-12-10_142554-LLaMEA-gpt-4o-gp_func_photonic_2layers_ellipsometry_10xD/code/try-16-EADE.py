import numpy as np

class EADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.elite_fraction = 0.05  # New elite retention parameter

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            elite_count = int(self.elite_fraction * self.population_size)  # Calculate elite count
            elite_indices = fitness.argsort()[:elite_count]  # Identify elite individuals
            new_pop = np.copy(pop)  # Initialize new population with current pop

            for i in range(self.population_size):
                if i in elite_indices:  # Preserve elite individuals
                    continue
                
                indices = list(range(0, i)) + list(range(i + 1, self.population_size))
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                perturbation = np.random.normal(scale=0.1, size=self.dim)  # New stochastic perturbation
                mutant = np.clip(a + self.mutation_factor * (b - c) + perturbation, lb, ub)
                
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fitness

                if eval_count >= self.budget:
                    break

            pop = new_pop  # Update population
            if eval_count % (self.population_size * 2) == 0:
                diversity = np.mean([np.linalg.norm(p1 - p2) for p1 in pop for p2 in pop]) / self.dim
                self.mutation_factor = 0.3 + 0.7 * diversity
                self.crossover_rate = 0.1 + 0.8 * (1 - diversity)

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]