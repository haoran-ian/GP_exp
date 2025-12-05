import numpy as np

class QIADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.bounds = (-5.0, 5.0)

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))

    def quantum_superposition(self, population):
        return np.mean(population, axis=0) + np.random.uniform(-1, 1, self.dim)

    def mutate(self, population, F):
        idxs = np.random.choice(range(self.population_size), 3, replace=False)
        a, b, c = population[idxs]
        return a + F * (b - c)

    def crossover(self, target, mutant, CR):
        cross_points = np.random.rand(self.dim) < CR
        return np.where(cross_points, mutant, target)

    def select(self, population, scores, trial, trial_score):
        if trial_score < scores[np.argmin(scores)]:
            idx = scores.argmax()
            population[idx] = trial
            scores[idx] = trial_score

    def __call__(self, func):
        population = self.initialize_population()
        scores = np.array([func(ind) for ind in population])
        evaluations = 0
        while evaluations < self.budget:
            for i, target in enumerate(population):
                if evaluations >= self.budget:
                    break
                F = np.random.uniform(0.4, 0.9)  # Dynamic F
                CR = np.random.uniform(0.8, 1.0) # Dynamic CR
                mutant = self.mutate(population, F)
                trial = self.crossover(target, mutant, CR)
                trial = np.clip(trial, *self.bounds)
                trial_score = func(trial)
                evaluations += 1
                self.select(population, scores, trial, trial_score)
            population = np.apply_along_axis(self.quantum_superposition, 0, population)
        best_idx = scores.argmin()
        return population[best_idx], scores[best_idx]