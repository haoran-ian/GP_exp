import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class EnhancedDynamicEvolutionaryStrategy:
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
        self.surrogate_model = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=1e-6)
        self.use_surrogate = True  # Enable surrogate model

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.fitness = np.full(self.population_size, np.inf)

    def evaluate_fitness(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])

    def mutate(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.bounds.lb, self.bounds.ub)
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def select(self, target_idx, trial_vector, trial_fitness):
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial_vector
            self.fitness[target_idx] = trial_fitness

    def adapt_parameters(self):
        self.crossover_rate = 0.1 + np.random.rand() * 0.9
        self.mutation_factor = 0.6 + np.random.rand() * 0.4

    def control_population_size(self):
        best_fitness = np.min(self.fitness)
        if self.dynamic_population_control and best_fitness < 1e-5:
            self.population_size = max(int(self.population_size * 0.9), self.dim * 2)

    def surrogate_evaluate(self, trial_vector):
        predicted_fitness, _ = self.surrogate_model.predict(trial_vector.reshape(1, -1), return_std=True)
        return predicted_fitness[0]

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population(self.bounds.lb, self.bounds.ub)
        self.evaluate_fitness(func)

        evaluations = self.population_size
        all_data_X = self.population.copy()
        all_data_y = self.fitness.copy()

        while evaluations < self.budget:
            self.adapt_parameters()
            self.control_population_size()
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                
                if self.use_surrogate and evaluations >= self.population_size:
                    trial_fitness = self.surrogate_evaluate(trial)
                else:
                    trial_fitness = func(trial)
                    evaluations += 1
                    if self.use_surrogate and evaluations % (self.population_size // 2) == 0:
                        self.surrogate_model.fit(all_data_X, all_data_y)
                    all_data_X = np.vstack((all_data_X, trial))
                    all_data_y = np.append(all_data_y, trial_fitness)

                self.select(i, trial, trial_fitness)
                if evaluations >= self.budget:
                    break

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]