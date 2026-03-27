import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Initial differential weight
        self.CR = 0.9  # Initial crossover probability
        self.greedy_probability = 0.1
        self.init_population()
        self.F_adapt_rate = 0.1
        self.CR_adapt_rate = 0.1

    def init_population(self):
        self.population = np.random.rand(self.population_size, self.dim)
        self.scores = np.full(self.population_size, np.inf)

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
        diversity_factor = 1 / (1 + np.std(self.scores))
        perturbation = np.random.uniform(-0.1, 0.1, self.dim) * diversity_factor
        greedy_candidate = np.clip(individual + perturbation, 0, 1)
        return greedy_candidate

    def adapt_parameters(self, success_rate):
        self.F = min(1.0, max(0.1, self.F + self.F_adapt_rate * (success_rate - 0.2)))
        self.CR = min(1.0, max(0.1, self.CR + self.CR_adapt_rate * (success_rate - 0.2)))

    def __call__(self, func):
        bounds = func.bounds
        successful_trials = 0
        total_trials = 0

        for i in range(self.population_size):
            self.population[i] = bounds.lb + (bounds.ub - bounds.lb) * self.population[i]
            self.scores[i] = func(self.population[i])
            self.budget -= 1

        while self.budget > 0:
            for i in range(self.population_size):
                donor = self.mutate(i)
                trial = self.crossover(self.population[i], donor)

                if np.random.rand() < self.greedy_probability:
                    trial = self.local_greedy_search(trial)

                trial_denormalized = bounds.lb + (bounds.ub - bounds.lb) * trial
                trial_score = func(trial_denormalized)
                self.budget -= 1
                total_trials += 1

                if trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score
                    successful_trials += 1

                if self.budget <= 0:
                    break

            success_rate = successful_trials / total_trials
            self.adapt_parameters(success_rate)

        best_idx = np.argmin(self.scores)
        best_solution = self.population[best_idx]
        return bounds.lb + (bounds.ub - bounds.lb) * best_solution