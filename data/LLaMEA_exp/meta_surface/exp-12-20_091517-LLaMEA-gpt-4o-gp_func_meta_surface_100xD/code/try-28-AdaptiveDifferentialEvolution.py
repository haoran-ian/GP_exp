import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 12 * dim
        self.pop = None
        self.fitness = None
        self.bounds = None
        self.F = 0.5
        self.CR = 0.9

    def levy_flight(self, lam=1.5):
        sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) / 
                 (np.math.gamma((1 + lam) / 2) * lam * 2**((lam - 1) / 2)))**(1 / lam)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        return u / abs(v)**(1 / lam)

    def stochastic_ranking(self, fitness, constraint_violations):
        return np.lexsort((fitness, constraint_violations))

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self.pop = np.random.rand(self.initial_population_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        self.fitness = np.array([func(ind) for ind in self.pop])
        evaluations = self.initial_population_size
        population_size = self.initial_population_size

        while evaluations < self.budget:
            constraint_violations = np.array([np.sum(ind < self.bounds[0]) + np.sum(ind > self.bounds[1]) for ind in self.pop])
            sorted_indices = self.stochastic_ranking(self.fitness, constraint_violations)
            self.pop = self.pop[sorted_indices]
            self.fitness = self.fitness[sorted_indices]

            new_population = []
            for i in range(population_size):
                candidates = list(range(population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)

                lev = self.levy_flight()
                mutant = np.clip(self.pop[a] + self.F * (self.pop[b] - self.pop[c]) + lev, self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.pop[i])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    new_population.append((trial, trial_fitness))
                else:
                    new_population.append((self.pop[i], self.fitness[i]))

                if evaluations >= self.budget:
                    break

            if evaluations < self.budget:
                self.pop, self.fitness = zip(*new_population)
                self.pop = np.array(self.pop)
                self.fitness = np.array(self.fitness)
                population_size = max(5, int(population_size * 0.95))  # adaptively reduce population size

            self.F = np.clip(self.F + np.random.uniform(-0.12, 0.12), 0.1, 0.9)
            self.CR = np.clip(self.CR + np.random.uniform(-0.1, 0.1), 0.1, 1.0)

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx]