import numpy as np

class DynamicPopulationDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 12 * dim
        self.pop = None
        self.fitness = None
        self.bounds = None
        self.F = 0.5
        self.CR = 0.9
        self.inertia_weight = 0.9
        self.chaotic_x = np.random.rand()

    def levy_flight(self, lam=1.5):
        sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) / 
                 (np.math.gamma((1 + lam) / 2) * lam * 2**((lam - 1) / 2)))**(1 / lam)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        return u / abs(v)**(1 / lam)

    def chaotic_sequence(self, method):
        if method == 'logistic':
            self.chaotic_x = 4.0 * self.chaotic_x * (1.0 - self.chaotic_x)
        elif method == 'tent':
            self.chaotic_x = 2.0 * self.chaotic_x if self.chaotic_x < 0.5 else 2.0 * (1.0 - self.chaotic_x)
        return self.chaotic_x

    def update_inertia_weight(self, eval_ratio):
        self.inertia_weight = 0.4 + 0.5 * (1 - eval_ratio)

    def dynamic_population_size(self, eval_ratio):
        min_pop_size = self.dim + 1
        max_pop_size = self.initial_population_size
        return int(min_pop_size + (max_pop_size - min_pop_size) * (1 - eval_ratio))

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self.pop = np.random.rand(self.initial_population_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        self.fitness = np.array([func(ind) for ind in self.pop])
        evaluations = self.initial_population_size

        while evaluations < self.budget:
            eval_ratio = evaluations / self.budget
            self.update_inertia_weight(eval_ratio)

            method = 'logistic' if np.random.rand() < 0.5 else 'tent'
            chaotic_factor = self.chaotic_sequence(method)
            adaptive_F = self.F * chaotic_factor * self.inertia_weight
            adaptive_CR = self.CR * chaotic_factor

            current_pop_size = self.dynamic_population_size(eval_ratio)
            self.pop = self.pop[:current_pop_size]
            self.fitness = self.fitness[:current_pop_size]

            for i in range(current_pop_size):
                candidates = list(range(current_pop_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)

                lev = self.levy_flight()
                mutant_levy = np.clip(self.pop[a] + adaptive_F * (self.pop[b] - self.pop[c]) + lev, self.bounds[0], self.bounds[1])
                mutant_classic = np.clip(self.pop[a] + adaptive_F * (self.pop[b] - self.pop[c]), self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dim) < adaptive_CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant_levy if np.random.rand() < 0.5 else mutant_classic, self.pop[i])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.pop[i] = trial

                if evaluations >= self.budget:
                    break

            diversity = np.std(self.pop, axis=0).mean()
            self.F = np.clip(self.F + np.random.uniform(-0.1, 0.1) * diversity, 0.1, 0.9)
            self.CR = np.clip(self.CR + np.random.uniform(-0.05, 0.05), 0.1, 1.0)

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx]