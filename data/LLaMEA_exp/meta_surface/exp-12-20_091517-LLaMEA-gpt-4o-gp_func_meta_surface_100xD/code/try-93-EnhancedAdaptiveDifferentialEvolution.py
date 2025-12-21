import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 15 * dim
        self.pop = None
        self.fitness = None
        self.bounds = None
        self.F = 0.5
        self.CR = 0.9
        self.inertia_weight = 0.9
        self.chaotic_x1 = np.random.rand()
        self.chaotic_x2 = np.random.rand()
        self.exploitation_ratio = 0.5
        self.mutation_adaptivity = 0.1

    def levy_flight(self, lam=1.5):
        sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) / 
                (np.math.gamma((1 + lam) / 2) * lam * 2**((lam - 1) / 2)))**(1 / lam)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        return u / abs(v)**(1 / lam)

    def chaotic_sequence(self):
        self.chaotic_x1 = 4.0 * self.chaotic_x1 * (1.0 - self.chaotic_x1)
        self.chaotic_x2 = 3.75 * self.chaotic_x2 * (1.0 - self.chaotic_x2)
        return (self.chaotic_x1 + self.chaotic_x2) / 2

    def update_inertia_weight(self, eval_ratio):
        self.inertia_weight = 0.4 + 0.5 * (1 - eval_ratio)

    def quantum_exploration(self, current, best):
        return current + np.random.uniform(-0.5, 0.5, size=self.dim) * (best - current) * self.exploitation_ratio

    def adaptive_chaotic_factor(self):
        chaotic_x = self.chaotic_sequence()
        if np.random.rand() < 0.15:
            chaotic_x = 1 - chaotic_x
        return chaotic_x

    def adapt_exploitation_ratio(self, diversity):
        self.exploitation_ratio = 0.4 + 0.6 * (1 - diversity)

    def adaptive_learning_rate(self, method='logistic'):
        return 0.5 * self.chaotic_sequence()

    def evaluate_population(self, func):
        return np.array([func(ind) for ind in self.pop])

    def update_mutation_adaptivity(self, diversity, improvement):
        self.mutation_adaptivity = 0.1 + 0.4 * (improvement / (diversity + 1e-6))

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self.pop = np.random.rand(self.population_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        self.fitness = self.evaluate_population(func)
        evaluations = self.population_size
        prev_best_fitness = np.min(self.fitness)

        while evaluations < self.budget:
            eval_ratio = evaluations / self.budget
            self.update_inertia_weight(eval_ratio)

            chaotic_factor = self.adaptive_chaotic_factor()
            adaptive_F = self.F * chaotic_factor * self.inertia_weight
            adaptive_CR = self.CR * chaotic_factor

            for i in range(self.population_size):
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)

                lev = self.levy_flight()
                mutant_levy = np.clip(self.pop[a] + adaptive_F * self.mutation_adaptivity * (self.pop[b] - self.pop[c]) + lev, self.bounds[0], self.bounds[1])
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

            best_idx = np.argmin(self.fitness)
            best_individual = self.pop[best_idx]
            best_fitness = self.fitness[best_idx]
            improvement = prev_best_fitness - best_fitness

            for i in range(self.population_size):
                self.pop[i] = self.quantum_exploration(self.pop[i], best_individual)

            diversity = np.std(self.pop, axis=0).mean()
            self.F = np.clip(self.F + self.adaptive_learning_rate() * diversity, 0.1, 0.9)
            self.CR = np.clip(self.CR + np.random.uniform(-0.025, 0.025), 0.1, 1.0)
            self.adapt_exploitation_ratio(diversity)
            self.update_mutation_adaptivity(diversity, improvement)
            prev_best_fitness = best_fitness

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx]