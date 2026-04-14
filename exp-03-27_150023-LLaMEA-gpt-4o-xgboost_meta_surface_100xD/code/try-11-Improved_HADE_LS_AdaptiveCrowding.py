import numpy as np

class Improved_HADE_LS_AdaptiveCrowding:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.population = None
        self.fitness = None
        self.CR = 0.9  # Crossover probability
        self.F = 0.8   # Differential weight
        self.evaluations = 0

    def init_population(self, bounds):
        lower_bound = bounds.lb
        upper_bound = bounds.ub
        self.population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def compute_diversity(self):
        mean_position = np.mean(self.population, axis=0)
        diversity = np.mean(np.linalg.norm(self.population - mean_position, axis=1))
        return diversity

    def adapt_parameters(self):
        diversity = self.compute_diversity()
        self.CR = 0.5 + 0.4 * (diversity / np.sqrt(self.dim))
        self.F = 0.5 + 0.3 * (diversity / np.sqrt(self.dim))

    def differential_evolution(self, target_idx, bounds):
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
        cross_points = np.random.rand(self.dim) < self.CR
        trial = np.where(cross_points, mutant, self.population[target_idx])
        return trial

    def local_search(self, cand, bounds):
        intensity = 0.1 * (1 - self.evaluations / self.budget)  # Adaptive intensity
        perturbation = np.clip(cand + np.random.normal(0, intensity, self.dim), bounds.lb, bounds.ub)
        return perturbation

    def crowding_distance_sort(self, trials, trial_fitness):
        if len(trial_fitness) == 0:
            return []
        distances = np.zeros(len(trials))
        for i in range(self.dim):
            sorted_indices = np.argsort(trials[:, i])
            sorted_fitness = trial_fitness[sorted_indices]
            max_fitness = np.max(sorted_fitness)
            if max_fitness - np.min(sorted_fitness) == 0:
                continue
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            for j in range(1, len(trials) - 1):
                distances[sorted_indices[j]] += (sorted_fitness[j+1] - sorted_fitness[j-1]) / (max_fitness - np.min(sorted_fitness))
        return np.argsort(-distances)

    def __call__(self, func):
        self.init_population(func.bounds)
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.population[np.argmin(self.fitness)]

        while self.evaluations < self.budget:
            self.adapt_parameters()  # Adapt parameters based on diversity

            trials = []
            trial_fitness = []
            for i in range(self.population_size):
                trial = self.differential_evolution(i, func.bounds)
                trial_fitness_value = func(trial)
                self.evaluations += 1

                if trial_fitness_value < self.fitness[i]:
                    trials.append(trial)
                    trial_fitness.append(trial_fitness_value)

                if np.random.rand() < 0.1:
                    local_candidate = self.local_search(self.population[i], func.bounds)
                    local_fitness = func(local_candidate)
                    self.evaluations += 1
                    if local_fitness < trial_fitness_value:
                        trials[-1] = local_candidate
                        trial_fitness[-1] = local_fitness

                if self.evaluations >= self.budget:
                    break

            sorted_indices = self.crowding_distance_sort(np.array(trials), np.array(trial_fitness))
            for idx in sorted_indices:
                self.population[idx] = trials[idx]
                self.fitness[idx] = trial_fitness[idx]

        return self.population[np.argmin(self.fitness)]