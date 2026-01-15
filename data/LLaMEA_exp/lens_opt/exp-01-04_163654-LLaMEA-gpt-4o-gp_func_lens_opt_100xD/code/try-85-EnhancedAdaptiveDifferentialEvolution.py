import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.num_subpopulations = 3
        self.subpop_size = self.pop_size // self.num_subpopulations
        self.F = 0.5
        self.CR = 0.9

    def chaotic_initialization(self, lb, ub, size):
        x = np.zeros(size)
        x[0] = np.random.rand()
        for i in range(1, size[0]):
            x[i] = 4 * x[i - 1] * (1 - x[i - 1])
        scaled_x = lb + (ub - lb) * x
        return scaled_x

    def self_adaptive_parameters(self):
        self.F = np.random.uniform(0.4, 0.9)
        self.CR = np.random.uniform(0.1, 0.9)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        populations = [self.chaotic_initialization(lb, ub, (self.subpop_size, self.dim))
                       for _ in range(self.num_subpopulations)]
        fitnesses = [np.array([func(ind) for ind in pop]) for pop in populations]
        num_evaluations = self.pop_size

        while num_evaluations < self.budget:
            for pop_idx, (population, fitness) in enumerate(zip(populations, fitnesses)):
                new_population = np.copy(population)
                for i in range(self.subpop_size):
                    self.self_adaptive_parameters()
                    idxs = [idx for idx in range(self.subpop_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.F * (b - c), lb, ub)
                    cross_points = np.random.rand(self.dim) < self.CR
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

                populations[pop_idx] = new_population
                fitnesses[pop_idx] = fitness

            # Merge populations periodically for global information sharing
            if num_evaluations % (self.pop_size * 5) == 0:
                combined_pop = np.vstack(populations)
                combined_fit = np.concatenate(fitnesses)
                best_indices = np.argsort(combined_fit)[:self.pop_size]
                for pop_idx in range(self.num_subpopulations):
                    populations[pop_idx] = combined_pop[best_indices[pop_idx*self.subpop_size: (pop_idx+1)*self.subpop_size]]
                    fitnesses[pop_idx] = combined_fit[best_indices[pop_idx*self.subpop_size: (pop_idx+1)*self.subpop_size]]

        best_idx = np.argmin([np.min(fit) for fit in fitnesses])
        best_pop_idx = np.argmin(fitnesses[best_idx])
        return populations[best_idx][best_pop_idx], fitnesses[best_idx][best_pop_idx]