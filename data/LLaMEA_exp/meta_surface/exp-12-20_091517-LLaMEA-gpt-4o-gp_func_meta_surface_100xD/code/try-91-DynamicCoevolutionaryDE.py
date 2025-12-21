import numpy as np

class DynamicCoevolutionaryDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 * dim
        self.pop = None
        self.fitness = None
        self.bounds = None
        self.F = 0.5
        self.CR = 0.9
        self.subpopulations = 5
        self.subpop_size = self.population_size // self.subpopulations
        self.diversity_threshold = 0.1

    def evaluate_population(self, func):
        return np.array([func(ind) for ind in self.pop])

    def coevolve_subpopulations(self, func):
        for subpop_idx in range(self.subpopulations):
            start = subpop_idx * self.subpop_size
            end = start + self.subpop_size
            subpop = self.pop[start:end]
            subfitness = self.fitness[start:end]

            for i in range(self.subpop_size):
                candidates = list(range(self.subpop_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)

                mutant_vector = self.mutate(subpop, a, b, c)
                trial_vector = self.crossover(subpop[i], mutant_vector)
                trial_vector = np.clip(trial_vector, self.bounds[0], self.bounds[1])

                trial_fitness = func(trial_vector)
                if trial_fitness < subfitness[i]:
                    subfitness[i] = trial_fitness
                    subpop[i] = trial_vector

            self.pop[start:end] = subpop
            self.fitness[start:end] = subfitness

    def mutate(self, subpop, a, b, c):
        return subpop[a] + self.F * (subpop[b] - subpop[c])

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)

    def adapt_parameters(self, diversity):
        self.F = np.clip(self.F + np.random.uniform(-0.05, 0.05), 0.1, 0.9)
        self.CR = np.clip(self.CR + np.random.uniform(-0.05, 0.05), 0.1, 1.0)

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self.pop = np.random.rand(self.population_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        self.fitness = self.evaluate_population(func)
        evaluations = self.population_size

        while evaluations < self.budget:
            self.coevolve_subpopulations(func)
            diversity = np.std(self.pop, axis=0).mean()
            self.adapt_parameters(diversity)

            if diversity < self.diversity_threshold:
                self.subpopulations = max(2, self.subpopulations - 1)

            evaluations += self.subpop_size * self.subpopulations
            if evaluations >= self.budget:
                break

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx]