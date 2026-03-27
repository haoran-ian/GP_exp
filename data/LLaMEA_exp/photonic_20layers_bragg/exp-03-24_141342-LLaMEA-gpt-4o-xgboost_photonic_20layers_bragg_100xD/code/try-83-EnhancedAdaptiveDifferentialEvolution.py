import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.init_population()
        self.local_search_intensity = 0.1
        self.adaptive_resizing_threshold = 0.2
        self.min_population_size = 4 * dim
        self.opposition_rate = 0.3
        self.levy_alpha = 1.5
        self.F_initial = 0.5
        self.CR_initial = 0.9
        self.F_adaptive_rate = 0.1
        self.CR_adaptive_rate = 0.1
        self.success_rate = np.zeros(self.population_size)
        self.failure_rate = np.zeros(self.population_size)
        self.diversity_threshold = 1e-5

    def init_population(self):
        logistic_map_r = 4.0
        self.population = np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            for j in range(self.dim):
                self.population[i, j] = logistic_map_r * self.population[i, j] * (1 - self.population[i, j])
        self.scores = np.full(self.population_size, np.inf)

    def mutate(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = self.adaptive_scaling_f(target_idx)
        donor_vector = self.population[a] + F * (self.population[b] - self.population[c])
        return np.clip(donor_vector, 0, 1)

    def crossover(self, target, donor, target_idx):
        CR = self.adaptive_scaling_cr(target_idx)
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

    def adaptive_local_greedy_search(self, individual):
        perturbation = np.random.uniform(-self.local_search_intensity, self.local_search_intensity, self.dim)
        greedy_candidate = np.clip(individual + perturbation, 0, 1)
        return greedy_candidate

    def adaptive_scaling_f(self, idx):
        return self.F_initial + self.F_adaptive_rate * (self.success_rate[idx] - self.failure_rate[idx])

    def adaptive_scaling_cr(self, idx):
        return self.CR_initial + self.CR_adaptive_rate * (self.success_rate[idx] - self.failure_rate[idx])

    def dynamic_population_resizing(self):
        if np.random.rand() < self.adaptive_resizing_threshold and self.population_size > self.min_population_size:
            self.population_size = max(self.min_population_size, int(self.population_size * 0.9))
            self.population = self.population[:self.population_size]
            self.scores = self.scores[:self.population_size]
            self.success_rate = self.success_rate[:self.population_size]
            self.failure_rate = self.failure_rate[:self.population_size]

    def calculate_diversity(self):
        centroid = np.mean(self.population, axis=0)
        diversity = np.mean(np.linalg.norm(self.population - centroid, axis=1))
        return diversity

    def __call__(self, func):
        bounds = func.bounds
        for i in range(self.population_size):
            self.population[i] = bounds.lb + (bounds.ub - bounds.lb) * self.population[i]
            self.scores[i] = func(self.population[i])
            self.budget -= 1

        while self.budget > 0:
            best_idx = np.argmin(self.scores)
            best_individual = self.population[best_idx]
            diversity = self.calculate_diversity()

            for i in range(self.population_size):
                donor = self.mutate(i)
                donor = self.levy_flight(donor)
                trial = self.crossover(self.population[i], donor, i)
                trial = self.opposition_based_learning(trial, best_individual)

                if np.random.rand() < self.local_search_intensity:
                    trial = self.adaptive_local_greedy_search(trial)

                trial_denormalized = bounds.lb + (bounds.ub - bounds.lb) * trial
                trial_score = func(trial_denormalized)
                self.budget -= 1

                if trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score
                    self.success_rate[i] += 1
                else:
                    self.failure_rate[i] += 1

                if self.budget <= 0:
                    break

            if diversity < self.diversity_threshold:
                perturbation = np.random.uniform(-0.1, 0.1, self.population.shape)
                self.population = np.clip(self.population + perturbation, 0, 1)

            self.dynamic_population_resizing()

        best_idx = np.argmin(self.scores)
        best_solution = self.population[best_idx]
        return bounds.lb + (bounds.ub - bounds.lb) * best_solution