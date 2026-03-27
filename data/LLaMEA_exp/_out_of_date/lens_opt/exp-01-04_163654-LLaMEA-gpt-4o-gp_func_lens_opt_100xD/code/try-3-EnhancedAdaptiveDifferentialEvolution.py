import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20
        self.pop_size = self.initial_pop_size
        self.F_min, self.F_max = 0.4, 0.9
        self.CR_min, self.CR_max = 0.1, 0.9
        self.F = np.random.uniform(self.F_min, self.F_max)
        self.CR = np.random.uniform(self.CR_min, self.CR_max)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size

        best_fitness = np.min(fitness)
        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
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
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        # Adapt F and CR
                        self.F = np.clip(self.F + np.random.uniform(-0.1, 0.1), self.F_min, self.F_max)
                        self.CR = np.clip(self.CR + np.random.uniform(-0.1, 0.1), self.CR_min, self.CR_max)
                if num_evaluations >= self.budget:
                    break

            population = new_population

            # Adaptive population resizing
            if num_evaluations < self.budget and num_evaluations % 100 == 0:
                performance = np.std(fitness) / np.abs(np.mean(fitness)) if np.mean(fitness) != 0 else 0
                if performance < 0.1 and self.pop_size > 10:
                    self.pop_size -= 1
                elif performance > 0.2 and self.pop_size < 50:
                    self.pop_size += 1
                new_pop = np.random.uniform(lb, ub, (self.pop_size - len(population), self.dim))
                new_fitness = np.array([func(ind) for ind in new_pop])
                num_evaluations += len(new_pop)
                population = np.vstack((population, new_pop))
                fitness = np.append(fitness, new_fitness)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]