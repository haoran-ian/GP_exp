import numpy as np

class ImprovedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20
        self.max_pop_size = 60
        self.min_pop_size = 10
        self.F_range = (0.4, 0.9)
        self.CR_range = (0.5, 1.0)
        self.adapt_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_pop_size
        population = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = pop_size

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = np.random.uniform(*self.F_range)
                CR = np.random.uniform(*self.CR_range)
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                num_evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                if num_evaluations >= self.budget:
                    break
            
            population = new_population

            # Adapt Population Size
            best_idx = np.argmin(fitness)
            best_fitness = fitness[best_idx]
            if num_evaluations / self.budget < 0.5:
                pop_size = min(self.max_pop_size, pop_size + self.adapt_rate * self.initial_pop_size)
            else:
                pop_size = max(self.min_pop_size, pop_size - self.adapt_rate * self.initial_pop_size)

            # Re-evaluate fitness for the adjusted population size
            if len(population) != int(pop_size):
                population = np.resize(population, (int(pop_size), self.dim))
                fitness = np.array([func(ind) for ind in population])
                num_evaluations += len(population)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]