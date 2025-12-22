import numpy as np

class AdaptiveMemeticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.strategies = [
            self.differential_evolution, 
            self.chaotic_local_search
        ]
        self.strategy_weights = np.ones(len(self.strategies)) / len(self.strategies)
        self.performance_history = np.zeros(len(self.strategies))

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            chosen_strategy_idx = self.select_strategy()
            chosen_strategy = self.strategies[chosen_strategy_idx]
            new_population, new_fitness = chosen_strategy(population, fitness, func, bounds)
            evals += len(new_fitness)
            
            if evals > self.budget:
                excess = evals - self.budget
                new_population = new_population[:-excess]
                new_fitness = new_fitness[:-excess]
                evals = self.budget

            combined_population = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population, fitness = combined_population[best_indices], combined_fitness[best_indices]

            self.update_strategy_weights(new_fitness, chosen_strategy_idx)

        return population[np.argmin(fitness)]

    def select_strategy(self):
        c = np.random.rand()
        chaotic_factor = (4 * c * (1 - c))  # Logistic map for stronger chaos
        chaos_strategy = np.argmax(chaotic_factor * self.strategy_weights)
        return chaos_strategy

    def update_strategy_weights(self, new_fitness, strategy_idx):
        improvement = np.maximum(0, np.min(new_fitness) - np.min(self.performance_history)) / (np.min(new_fitness) + 1e-6)
        self.performance_history[strategy_idx] += improvement
        self.strategy_weights = self.performance_history / self.performance_history.sum() + 1e-6

    def differential_evolution(self, population, fitness, func, bounds):
        F = 0.5 + 0.5 * np.random.rand() * (1 - np.mean(fitness) / np.max(fitness))
        CR = 0.9
        new_population = np.empty_like(population)
        new_fitness = np.empty(self.population_size)
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            x1, x2, x3 = population[indices]
            mutant = np.clip(x1 + F * (x2 - x3), bounds[:, 0], bounds[:, 1])
            crossover = np.random.rand(self.dim) < CR
            trial = np.where(crossover, mutant, population[i])
            new_population[i] = trial
            new_fitness[i] = func(trial)
        return new_population, new_fitness

    def chaotic_local_search(self, population, fitness, func, bounds):
        new_population = np.empty_like(population)
        new_fitness = np.empty(self.population_size)
        for i in range(self.population_size):
            chaotic_sequence = np.random.rand(self.dim)
            for j in range(3):  # Reduced iterations for faster convergence
                chaotic_sequence = np.sin(np.pi * chaotic_sequence)
                perturbation = (chaotic_sequence - 0.5) * 2 * (bounds[:, 1] - bounds[:, 0]) * 0.01
                trial = np.clip(population[i] + perturbation, bounds[:, 0], bounds[:, 1])
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    new_population[i], new_fitness[i] = trial, trial_fitness
                else:
                    new_population[i], new_fitness[i] = population[i], fitness[i]
        return new_population, new_fitness