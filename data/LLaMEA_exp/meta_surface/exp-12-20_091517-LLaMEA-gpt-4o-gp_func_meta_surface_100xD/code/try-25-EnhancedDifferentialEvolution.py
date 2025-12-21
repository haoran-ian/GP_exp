import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim  # Increased population size from 10 * dim to 12 * dim
        self.pop = None
        self.fitness = None
        self.bounds = None
        self.F = 0.6  # Modified mutation factor from 0.5 to 0.6
        self.CR = 0.9  # Initial crossover rate

    def levy_flight(self, lam=1.5):
        sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) / 
                (np.math.gamma((1 + lam) / 2) * lam * 2**((lam - 1) / 2)))**(1 / lam)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        return u / abs(v)**(1 / lam)

    def __call__(self, func):
        # Initialization
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self.pop = np.random.rand(self.population_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        self.fitness = np.array([func(ind) for ind in self.pop])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Select three distinct vectors
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)

                # Mutation with Levy Flight
                lev = self.levy_flight()
                mutant = np.clip(self.pop[a] + self.F * (self.pop[b] - self.pop[c]) + lev, self.bounds[0], self.bounds[1])

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

            # Dynamic adjustment of F and CR
            self.F = np.clip(self.F + np.random.uniform(-0.12, 0.12), 0.1, 0.9)  # Slightly increased range
            self.CR = np.clip(self.CR + np.random.uniform(-0.1, 0.1), 0.1, 1.0)

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx]