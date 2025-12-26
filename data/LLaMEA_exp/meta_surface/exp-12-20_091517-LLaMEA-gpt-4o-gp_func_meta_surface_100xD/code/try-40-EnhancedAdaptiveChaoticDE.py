import numpy as np

class EnhancedAdaptiveChaoticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim
        self.pop = None
        self.fitness = None
        self.bounds = None
        self.F = 0.5
        self.CR = 0.9
        self.elite_archive = []

    def levy_flight(self, lam=1.5):
        sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) / 
                (np.math.gamma((1 + lam) / 2) * lam * 2**((lam - 1) / 2)))**(1 / lam)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        return u / abs(v)**(1 / lam)

    def chaotic_sequence(self, x):
        return 4.0 * x * (1.0 - x)

    def opposition_based_learning(self):
        opposite_pop = self.bounds[0] + self.bounds[1] - self.pop
        opposite_fitness = np.array([func(ind) for ind in opposite_pop])
        return opposite_pop, opposite_fitness

    def update_archive(self, trial, trial_fitness):
        if len(self.elite_archive) < self.population_size:
            self.elite_archive.append((trial, trial_fitness))
        else:
            worst_index = np.argmax([fit for _, fit in self.elite_archive])
            if trial_fitness < self.elite_archive[worst_index][1]:
                self.elite_archive[worst_index] = (trial, trial_fitness)
    
    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self.pop = np.random.rand(self.population_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        self.fitness = np.array([func(ind) for ind in self.pop])
        evaluations = self.population_size

        chaotic_factor = np.random.rand()
        
        while evaluations < self.budget:
            chaotic_factor = self.chaotic_sequence(chaotic_factor)
            adaptive_F = self.F * chaotic_factor
            adaptive_CR = self.CR * chaotic_factor

            # Perform opposition-based learning
            opposite_pop, opposite_fitness = self.opposition_based_learning()
            for i in range(self.population_size):
                if opposite_fitness[i] < self.fitness[i]:
                    self.pop[i] = opposite_pop[i]
                    self.fitness[i] = opposite_fitness[i]

            for i in range(self.population_size):
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)

                lev = self.levy_flight()
                mutant = np.clip(self.pop[a] + adaptive_F * (self.pop[b] - self.pop[c]) + lev, self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dim) < adaptive_CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.pop[i])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.pop[i] = trial
                    self.update_archive(trial, trial_fitness)

                if evaluations >= self.budget:
                    break

            self.F = np.clip(self.F + np.random.uniform(-0.1, 0.1), 0.1, 0.9)
            self.CR = np.clip(self.CR + np.random.uniform(-0.05, 0.05), 0.1, 1.0)

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx]