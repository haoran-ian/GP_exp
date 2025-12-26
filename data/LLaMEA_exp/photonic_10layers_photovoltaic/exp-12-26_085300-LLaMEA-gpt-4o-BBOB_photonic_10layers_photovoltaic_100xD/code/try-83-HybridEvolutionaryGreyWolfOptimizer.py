import numpy as np

class HybridEvolutionaryGreyWolfOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.a_param = 2.0  # Grey Wolf parameter
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.population = None

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def adapt_parameters(self, evaluations):
        progress = evaluations / self.budget
        self.mutation_factor = 0.5 + 0.3 * (1 - progress)
        self.crossover_probability = 0.9 - 0.5 * (1 - progress)
        self.a_param = 2.0 * (1 - progress)  # A decreasing factor for Grey Wolf

    def differential_evolution_step(self, idx, population, bounds):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant_vector = np.clip(a + self.mutation_factor * (b - c), bounds.lb, bounds.ub)
        
        crossover_mask = np.random.rand(self.dim) < self.crossover_probability
        trial_vector = np.where(crossover_mask, mutant_vector, population[idx])
        return trial_vector

    def grey_wolf_update(self, population, alpha, beta, delta):
        A1 = 2 * self.a_param * np.random.rand(self.dim) - self.a_param
        C1 = 2 * np.random.rand(self.dim)
        D_alpha = np.abs(C1 * alpha - population)
        X1 = alpha - A1 * D_alpha

        A2 = 2 * self.a_param * np.random.rand(self.dim) - self.a_param
        C2 = 2 * np.random.rand(self.dim)
        D_beta = np.abs(C2 * beta - population)
        X2 = beta - A2 * D_beta

        A3 = 2 * self.a_param * np.random.rand(self.dim) - self.a_param
        C3 = 2 * np.random.rand(self.dim)
        D_delta = np.abs(C3 * delta - population)
        X3 = delta - A3 * D_delta

        return (X1 + X2 + X3) / 3

    def __call__(self, func):
        self.population = self.initialize_population(func.bounds)
        fitness = np.array([func(ind) for ind in self.population])
        evaluations = self.population_size

        while evaluations < self.budget:
            self.adapt_parameters(evaluations)
            alpha_idx = np.argmin(fitness)
            alpha = self.population[alpha_idx]
            beta_idx = np.argsort(fitness)[1]
            beta = self.population[beta_idx]
            delta_idx = np.argsort(fitness)[2]
            delta = self.population[delta_idx]

            new_population = np.empty_like(self.population)
            for i in range(self.population_size):
                if np.random.rand() < 0.5:
                    trial_vector = self.differential_evolution_step(i, self.population, func.bounds)
                else:
                    trial_vector = self.grey_wolf_update(self.population[i], alpha, beta, delta)

                trial_vector = np.clip(trial_vector, func.bounds.lb, func.bounds.ub)
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial_vector
                    fitness[i] = trial_fitness
                else:
                    new_population[i] = self.population[i]

                if evaluations >= self.budget:
                    break

            self.population = new_population

        best_idx = np.argmin(fitness)
        return self.population[best_idx]