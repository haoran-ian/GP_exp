import numpy as np

class AdvancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.init_population()
        self.adaptive_resizing_threshold = 0.2
        self.min_population_size = 4 * dim
        self.opposition_rate = 0.3
        self.levy_alpha = 1.5
        self.F_initial = 0.5
        self.CR_initial = 0.9
        self.F_adaptive_rate = 0.1
        self.CR_adaptive_rate = 0.1
        self.rank_probability = 0.2  # Probability for stochastic ranking

    def init_population(self):
        chaotic_sequence = np.linspace(0, 1, self.population_size)
        np.random.shuffle(chaotic_sequence)
        self.population = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            for j in range(self.dim):
                self.population[i, j] = 4.0 * chaotic_sequence[i] * (1 - chaotic_sequence[i])
        self.scores = np.full(self.population_size, np.inf)

    def mutate(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = self.adaptive_scaling_f()
        donor_vector = self.population[a] + F * (self.population[b] - self.population[c])
        return np.clip(donor_vector, 0, 1)

    def crossover(self, target, donor):
        CR = self.adaptive_scaling_cr()
        trial = np.copy(target)
        crossover_points = np.random.rand(self.dim) < CR
        trial[crossover_points] = donor[crossover_points]
        return trial

    def levy_flight(self, individual):
        step = np.random.standard_cauchy(self.dim)
        levy_step = step / np.power(np.abs(step), 1/self.levy_alpha)
        new_individual = individual + levy_step
        return np.clip(new_individual, 0, 1)

    def opposition_based_learning(self, individual, best_individual):
        if np.random.rand() < self.opposition_rate:
            return np.clip(1.0 - best_individual + individual, 0, 1)
        return individual

    def adaptive_scaling_f(self):
        return self.F_initial + np.random.uniform(-self.F_adaptive_rate, self.F_adaptive_rate)

    def adaptive_scaling_cr(self):
        return self.CR_initial + np.random.uniform(-self.CR_adaptive_rate, self.CR_adaptive_rate)

    def neighborhood_search(self, target):
        neighborhood_size = 0.05
        neighbor = target + np.random.uniform(-neighborhood_size, neighborhood_size, self.dim)
        return np.clip(neighbor, 0, 1)

    def stochastic_ranking(self, population, scores):
        sorted_indices = np.argsort(scores)
        for i in range(len(scores)):
            if np.random.rand() < self.rank_probability:
                j = np.random.choice(len(scores))
                if scores[i] < scores[j]:
                    sorted_indices[i], sorted_indices[j] = sorted_indices[j], sorted_indices[i]
        return sorted_indices

    def __call__(self, func):
        bounds = func.bounds
        for i in range(self.population_size):
            self.population[i] = bounds.lb + (bounds.ub - bounds.lb) * self.population[i]
            self.scores[i] = func(self.population[i])
            self.budget -= 1

        while self.budget > 0:
            sorted_indices = self.stochastic_ranking(self.population, self.scores)
            best_idx = sorted_indices[0]
            best_individual = self.population[best_idx]

            for i in range(self.population_size):
                donor = self.mutate(i)
                donor = self.levy_flight(donor)
                trial = self.crossover(self.population[i], donor)
                trial = self.opposition_based_learning(trial, best_individual)
                trial = self.neighborhood_search(trial)

                trial_denormalized = bounds.lb + (bounds.ub - bounds.lb) * trial
                trial_score = func(trial_denormalized)
                self.budget -= 1

                if trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

                if self.budget <= 0:
                    break

            self.dynamic_population_resizing()

        best_idx = np.argmin(self.scores)
        best_solution = self.population[best_idx]
        return bounds.lb + (bounds.ub - bounds.lb) * best_solution