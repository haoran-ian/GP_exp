import numpy as np

class EnhancedChaoticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim
        self.sub_populations = 3
        self.pop = [None] * self.sub_populations
        self.fitness = [None] * self.sub_populations
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

    def quantum_exploration(self, current, best):
        return current + np.random.uniform(-0.5, 0.5, size=self.dim) * (best - current)

    def adaptive_chaotic_factor(self, method):
        chaotic_x = self.chaotic_sequence(method)
        if np.random.rand() < 0.2:
            chaotic_x = 1 - chaotic_x
        return chaotic_x

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        sub_pop_size = self.population_size // self.sub_populations
        for k in range(self.sub_populations):
            self.pop[k] = np.random.rand(sub_pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
            self.fitness[k] = np.array([func(ind) for ind in self.pop[k]])

        evaluations = self.population_size

        while evaluations < self.budget:
            eval_ratio = evaluations / self.budget
            self.update_inertia_weight(eval_ratio)

            method = np.random.choice(['logistic', 'tent'])
            chaotic_factor = self.adaptive_chaotic_factor(method)
            adaptive_F = self.F * chaotic_factor * self.inertia_weight
            adaptive_CR = self.CR * chaotic_factor

            for k in range(self.sub_populations):
                for i in range(sub_pop_size):
                    candidates = list(range(sub_pop_size))
                    candidates.remove(i)
                    a, b, c = np.random.choice(candidates, 3, replace=False)
                    
                    lev = self.levy_flight()
                    mutant_levy = np.clip(self.pop[k][a] + adaptive_F * (self.pop[k][b] - self.pop[k][c]) + lev, self.bounds[0], self.bounds[1])
                    mutant_classic = np.clip(self.pop[k][a] + adaptive_F * (self.pop[k][b] - self.pop[k][c]), self.bounds[0], self.bounds[1])

                    cross_points = np.random.rand(self.dim) < adaptive_CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant_levy if np.random.rand() < 0.5 else mutant_classic, self.pop[k][i])

                    trial_fitness = func(trial)
                    evaluations += 1

                    if trial_fitness < self.fitness[k][i]:
                        self.fitness[k][i] = trial_fitness
                        self.pop[k][i] = trial

                    if evaluations >= self.budget:
                        break

                best_idx = np.argmin(self.fitness[k])
                best_individual = self.pop[k][best_idx]

                for i in range(sub_pop_size):
                    self.pop[k][i] = self.quantum_exploration(self.pop[k][i], best_individual)

            for k in range(self.sub_populations):
                diversity = np.std(self.pop[k], axis=0).mean()
                self.F = np.clip(self.F + np.random.uniform(-0.1, 0.1) * diversity, 0.1, 0.9)
                self.CR = np.clip(self.CR + np.random.uniform(-0.05, 0.05), 0.1, 1.0)

        best_fitness = float('inf')
        best_solution = None
        for k in range(self.sub_populations):
            best_idx = np.argmin(self.fitness[k])
            if self.fitness[k][best_idx] < best_fitness:
                best_fitness = self.fitness[k][best_idx]
                best_solution = self.pop[k][best_idx]

        return best_solution