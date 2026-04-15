import numpy as np

class Enhanced_HADE_LS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.population = None
        self.fitness = None
        self.CR = 0.9  # Initial crossover probability
        self.F = 0.8   # Initial differential weight
        self.evaluations = 0
        self.noise_threshold = 1e-8

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

    def select_leader(self):
        sorted_indices = np.argsort(self.fitness)
        leader_index = sorted_indices[0]
        return self.population[leader_index]

    def differential_evolution(self, target_idx, bounds):
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
        cross_points = np.random.rand(self.dim) < self.CR
        trial = np.where(cross_points, mutant, self.population[target_idx])
        return trial

    def handle_noise(self, fitness):
        return fitness + np.random.uniform(-self.noise_threshold, self.noise_threshold)

    def local_search(self, cand, bounds):
        intensity = 0.1 * (1 - self.evaluations / self.budget)
        perturbation = np.clip(cand + np.random.normal(0, intensity, self.dim), bounds.lb, bounds.ub)
        return perturbation

    def __call__(self, func):
        self.init_population(func.bounds)
        for i in range(self.population_size):
            self.fitness[i] = self.handle_noise(func(self.population[i]))
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.population[np.argmin(self.fitness)]

        while self.evaluations < self.budget:
            self.adapt_parameters()

            leader = self.select_leader()

            for i in range(self.population_size):
                trial = self.differential_evolution(i, func.bounds)
                trial_fitness = self.handle_noise(func(trial))
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                if np.random.rand() < 0.1:
                    local_candidate = self.local_search(self.population[i], func.bounds)
                    local_fitness = self.handle_noise(func(local_candidate))
                    self.evaluations += 1
                    if local_fitness < self.fitness[i]:
                        self.population[i] = local_candidate
                        self.fitness[i] = local_fitness

                if self.evaluations >= self.budget:
                    break

        return self.population[np.argmin(self.fitness)]