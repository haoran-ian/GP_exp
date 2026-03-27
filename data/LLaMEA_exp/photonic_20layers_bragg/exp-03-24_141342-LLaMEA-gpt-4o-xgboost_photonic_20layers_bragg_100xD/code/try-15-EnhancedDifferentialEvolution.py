import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.greedy_probability = 0.1
        self.adaptation_rate = 0.1
        self.init_population()

    def init_population(self):
        self.population = np.random.rand(self.population_size, self.dim)
        self.scores = np.full(self.population_size, np.inf)

    def adapt_parameters(self):
        self.F = np.clip(self.F + self.adaptation_rate * (np.random.rand() - 0.5), 0.1, 0.9)
        self.greedy_probability = np.clip(self.greedy_probability + self.adaptation_rate * (np.random.rand() - 0.5), 0.05, 0.5)

    def mutate(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        donor_vector = (
            self.population[a] + self.F * (self.population[b] - self.population[c])
        )
        return np.clip(donor_vector, 0, 1)

    def crossover(self, target, donor):
        trial = np.copy(target)
        crossover_points = np.random.rand(self.dim) < self.CR
        trial[crossover_points] = donor[crossover_points]
        return trial

    def local_greedy_search(self, individual):
        perturbation = np.random.uniform(-0.1, 0.1, self.dim)
        greedy_candidate = np.clip(individual + perturbation, 0, 1)
        return greedy_candidate

    def __call__(self, func):
        bounds = func.bounds
        for i in range(self.population_size):
            self.population[i] = bounds.lb + (bounds.ub - bounds.lb) * self.population[i]
            self.scores[i] = func(self.population[i])
            self.budget -= 1

        while self.budget > 0:
            for i in range(self.population_size):
                self.adapt_parameters()
                donor = self.mutate(i)
                trial = self.crossover(self.population[i], donor)

                if np.random.rand() < self.greedy_probability:
                    trial = self.local_greedy_search(trial)

                trial_denormalized = bounds.lb + (bounds.ub - bounds.lb) * trial
                trial_score = func(trial_denormalized)
                self.budget -= 1

                if trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

                if self.budget <= 0:
                    break

        best_idx = np.argmin(self.scores)
        best_solution = self.population[best_idx]
        return bounds.lb + (bounds.ub - bounds.lb) * best_solution