import numpy as np

class EnhancedAdaptiveMemeticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.population = None
        self.bounds = None
        self.fitness = None
        self.crossover_rate = 0.5
        self.mutation_factor = 0.8
        self.generations = 0
        self.dynamic_population_control = True
        self.memory = []
        self.strategy_memory = {'DE/rand/1/bin': 0, 'DE/best/1/bin': 0, 'DE/cma-es': 0}
        self.cma_covariance = np.eye(dim)
        self.cma_mean = None

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.fitness = np.full(self.population_size, np.inf)
        self.cma_mean = np.mean(self.population, axis=0)

    def evaluate_fitness(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])

    def mutate(self, target_idx, strategy):
        indices = list(range(self.population_size))
        indices.remove(target_idx)

        if strategy == 'DE/rand/1/bin':
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        elif strategy == 'DE/best/1/bin':
            best_idx = np.argmin(self.fitness)
            a, b = np.random.choice(indices, 2, replace=False)
            mutant = self.population[best_idx] + self.mutation_factor * (self.population[a] - self.population[b])
        else:  # DE/cma-es
            deviation = np.random.multivariate_normal(np.zeros(self.dim), self.cma_covariance)
            mutant = self.cma_mean + deviation

        mutant = np.clip(mutant, self.bounds.lb, self.bounds.ub)
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def select(self, target_idx, trial_vector, trial_fitness, strategy):
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial_vector
            self.fitness[target_idx] = trial_fitness
            self.memory.append((self.crossover_rate, self.mutation_factor))
            self.strategy_memory[strategy] += 1

    def update_cma_parameters(self):
        weights = np.exp(-np.arange(1, self.population_size + 1) / (0.2 * self.population_size))
        weights /= np.sum(weights)
        ranked_indices = np.argsort(self.fitness)
        self.cma_mean = np.dot(weights, self.population[ranked_indices[:self.population_size]])
        cov_update = sum(
            w * np.outer(self.population[i] - self.cma_mean, self.population[i] - self.cma_mean)
            for i, w in zip(ranked_indices, weights)
        )
        self.cma_covariance = 0.9 * self.cma_covariance + 0.1 * cov_update

    def adapt_parameters(self):
        if self.memory:
            average_cr = np.mean([cr for cr, _ in self.memory])
            average_f = np.mean([mf for _, mf in self.memory])
            self.crossover_rate = np.clip(average_cr + np.random.rand() * 0.1, 0.1, 0.9)
            self.mutation_factor = np.clip(average_f + np.random.rand() * 0.1, 0.5, 1.0)
        else:
            self.crossover_rate = 0.1 + np.random.rand() * 0.9
            self.mutation_factor = 0.6 + np.random.rand() * 0.4

    def control_population_size(self):
        best_fitness = np.min(self.fitness)
        if self.dynamic_population_control and best_fitness < 1e-5:
            self.population_size = max(int(self.population_size * 0.9), self.dim * 2)

    def select_strategy(self):
        probabilities = np.array([self.strategy_memory['DE/rand/1/bin'], self.strategy_memory['DE/best/1/bin'], self.strategy_memory['DE/cma-es']])
        probabilities = probabilities / probabilities.sum()
        return np.random.choice(['DE/rand/1/bin', 'DE/best/1/bin', 'DE/cma-es'], p=probabilities)

    def local_search(self, candidate, func):
        perturbation = (np.random.rand(self.dim) - 0.5) * 0.1 * (self.bounds.ub - self.bounds.lb)
        neighbor = candidate + perturbation
        neighbor = np.clip(neighbor, self.bounds.lb, self.bounds.ub)
        neighbor_fitness = func(neighbor)
        return neighbor, neighbor_fitness

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population(self.bounds.lb, self.bounds.ub)
        self.evaluate_fitness(func)

        evaluations = self.population_size
        while evaluations < self.budget:
            self.adapt_parameters()
            self.control_population_size()
            for i in range(self.population_size):
                strategy = self.select_strategy()
                mutant = self.mutate(i, strategy)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1
                self.select(i, trial, trial_fitness, strategy)
                if evaluations >= self.budget:
                    break

            self.update_cma_parameters()

            best_idx = np.argmin(self.fitness)
            local_candidate, local_fitness = self.local_search(self.population[best_idx], func)
            if local_fitness < self.fitness[best_idx]:
                self.population[best_idx] = local_candidate
                self.fitness[best_idx] = local_fitness

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]