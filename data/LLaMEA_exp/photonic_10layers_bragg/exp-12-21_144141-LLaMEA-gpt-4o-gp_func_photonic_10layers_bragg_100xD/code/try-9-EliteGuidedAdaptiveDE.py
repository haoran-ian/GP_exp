import numpy as np

class EliteGuidedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.population = None
        self.fitness = None
        self.best_individual = None
        self.best_fitness = float('inf')
        self.bounds = None
        self.strategy_probabilities = [0.5, 0.5]  # Probabilities for DE/rand/1 and DE/best/1
        self.mutation_factors = [0.6, 0.9]  # Mutation factors for the strategies
        self.crossover_probability = 0.9
        self.success_history = [0, 0]  # Track successful applications of strategies

    def initialize_population(self):
        self.population = np.random.uniform(
            self.bounds.lb, self.bounds.ub, (self.pop_size, self.dim)
        )
        self.fitness = np.full(self.pop_size, float('inf'))

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.fitness[i] == float('inf'):
                self.fitness[i] = func(self.population[i])
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_individual = self.population[i].copy()

    def mutate(self, target_idx):
        indices = list(range(self.pop_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        strategy = np.random.choice(
            [0, 1], p=self.strategy_probabilities
        )
        
        if strategy == 0:  # DE/rand/1
            mutant = self.population[a] + self.mutation_factors[0] * (self.population[b] - self.population[c])
        else:  # DE/best/1
            mutant = self.best_individual + self.mutation_factors[1] * (self.population[b] - self.population[c])
        
        mutant = np.clip(mutant, self.bounds.lb, self.bounds.ub)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < max(0.5, self.crossover_probability * (self.best_fitness / np.median(self.fitness)))
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def adapt_strategy_probabilities(self, success, strategy):
        self.success_history[strategy] += success
        if success:
            self.strategy_probabilities = [
                min(p + 0.1, 1.0) if idx == strategy else max(p - 0.1, 0.0)
                for idx, p in enumerate(self.strategy_probabilities)
            ]
        else:
            self.strategy_probabilities = [
                max(p - 0.1, 0.0) if idx == strategy else min(p + 0.1, 1.0)
                for idx, p in enumerate(self.strategy_probabilities)
            ]
        total = sum(self.strategy_probabilities)
        self.strategy_probabilities = [p / total for p in self.strategy_probabilities]

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population()
        self.evaluate_population(func)
        
        evaluations = self.pop_size
        while evaluations < self.budget:
            for i in range(self.pop_size):
                mutant = self.mutate(i)
                strategy = np.random.choice([0, 1], p=self.strategy_probabilities)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_individual = trial.copy()
                    self.adapt_strategy_probabilities(True, strategy)
                else:
                    self.adapt_strategy_probabilities(False, strategy)

                if evaluations >= self.budget:
                    break

        return self.best_individual, self.best_fitness