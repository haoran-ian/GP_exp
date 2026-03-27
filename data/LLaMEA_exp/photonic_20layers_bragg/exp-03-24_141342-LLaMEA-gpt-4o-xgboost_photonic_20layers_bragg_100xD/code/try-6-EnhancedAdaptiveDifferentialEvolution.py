import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_base = 0.5  # Base differential weight
        self.CR_base = 0.9  # Base crossover probability
        self.greedy_probability = 0.1
        self.init_population()

    def init_population(self):
        self.population = np.random.rand(self.population_size, self.dim)
        self.scores = np.full(self.population_size, np.inf)
        self.F = np.full(self.population_size, self.F_base)
        self.CR = np.full(self.population_size, self.CR_base)

    def mutate(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        donor_vector = (
            self.population[a] + self.F[target_idx] * (self.population[b] - self.population[c])
        )
        return np.clip(donor_vector, 0, 1)

    def crossover(self, target, donor, target_idx):
        trial = np.copy(target)
        crossover_points = np.random.rand(self.dim) < self.CR[target_idx]
        trial[crossover_points] = donor[crossover_points]
        return trial

    def local_greedy_search(self, individual):
        perturbation = np.random.uniform(-0.1, 0.1, self.dim)
        greedy_candidate = np.clip(individual + perturbation, 0, 1)
        return greedy_candidate

    def stochastic_hill_climbing(self, individual, score, func):
        perturbation = np.random.normal(0, 0.05, self.dim)
        candidate = np.clip(individual + perturbation, 0, 1)
        candidate_score = func(candidate)
        self.budget -= 1
        if candidate_score < score:
            return candidate, candidate_score
        else:
            return individual, score

    def update_parameters(self, idx, success):
        if success:
            self.F[idx] = min(2.0, self.F[idx] * 1.2)
            self.CR[idx] = min(1.0, self.CR[idx] + 0.1)
        else:
            self.F[idx] = max(0.1, self.F[idx] * 0.9)
            self.CR[idx] = max(0.0, self.CR[idx] - 0.1)

    def __call__(self, func):
        bounds = func.bounds
        for i in range(self.population_size):
            self.population[i] = bounds.lb + (bounds.ub - bounds.lb) * self.population[i]
            self.scores[i] = func(self.population[i])
            self.budget -= 1

        while self.budget > 0:
            for i in range(self.population_size):
                donor = self.mutate(i)
                trial = self.crossover(self.population[i], donor, i)

                if np.random.rand() < self.greedy_probability:
                    trial = self.local_greedy_search(trial)

                trial_denormalized = bounds.lb + (bounds.ub - bounds.lb) * trial
                trial_score = func(trial_denormalized)
                self.budget -= 1

                if trial_score < self.scores[i]:
                    self.population[i], self.scores[i] = self.stochastic_hill_climbing(trial, trial_score, func)
                    self.update_parameters(i, True)
                else:
                    self.update_parameters(i, False)

                if self.budget <= 0:
                    break

        best_idx = np.argmin(self.scores)
        best_solution = self.population[best_idx]
        return bounds.lb + (bounds.ub - bounds.lb) * best_solution