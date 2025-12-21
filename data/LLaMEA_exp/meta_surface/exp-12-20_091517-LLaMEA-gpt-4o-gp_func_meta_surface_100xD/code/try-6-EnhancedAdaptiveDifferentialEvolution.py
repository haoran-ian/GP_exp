import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.pop = None
        self.fitness = None
        self.bounds = None
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9  # Initial crossover rate
        self.learning_rate = 0.1  # Rate to adjust F and CR

    def __call__(self, func):
        # Initialization
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self.pop = np.random.rand(self.population_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        self.fitness = np.array([func(ind) for ind in self.pop])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Select three distinct vectors with diversity check
                candidates = list(range(self.population_size))
                candidates.remove(i)
                selected = np.random.choice(candidates, 3, replace=False)
                a, b, c = selected if np.std(self.pop[selected], axis=0).mean() > 0.1 else np.random.choice(candidates, 3, replace=False)

                # Mutation
                mutant = np.clip(self.pop[a] + self.F * (self.pop[b] - self.pop[c]), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.pop[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.pop[i] = trial

                if evaluations >= self.budget:
                    break

            # Adaptive adjustment of F and CR based on population improvement
            mean_fitness = np.mean(self.fitness)
            F_improvement = self.learning_rate * (np.min(self.fitness) - mean_fitness)
            CR_improvement = self.learning_rate * (mean_fitness - np.min(self.fitness))
            self.F = np.clip(self.F + F_improvement, 0.1, 0.9)
            self.CR = np.clip(self.CR + CR_improvement, 0.1, 1.0)

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx]