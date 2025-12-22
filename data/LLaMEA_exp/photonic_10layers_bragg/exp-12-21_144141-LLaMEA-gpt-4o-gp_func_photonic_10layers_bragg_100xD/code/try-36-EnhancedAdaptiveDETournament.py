import numpy as np

class EnhancedAdaptiveDETournament:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.population = None
        self.fitness = None
        self.best_individual = None
        self.best_fitness = float('inf')
        self.bounds = None
        self.strategy_probabilities = [0.33, 0.33, 0.34]
        self.mutation_factors = [0.5, 0.7, 1.0]
        self.crossover_probability = 0.9
        self.success_history = [0, 0, 0]
        self.dynamic_adjustment_rate = 0.05
        self.dynamic_crossover_rate = 0.02
        self.dynamic_mutation_adjustment = 0.01
        self.elite_preservation_rate = 0.1
        self.fitness_decay_rate = 0.99  # New: decay rate for fitness-based adaptation

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
        strategy = np.random.choice([0, 1, 2], p=self.strategy_probabilities)

        if strategy == 0:
            mutant = self.population[a] + self.mutation_factors[0] * (self.population[b] - self.population[c])
        elif strategy == 1:
            mutant = self.best_individual + self.mutation_factors[1] * (self.population[b] - self.population[c])
        else:
            # Improve selection by using the best among random selections
            d = np.random.choice(indices)
            e = np.random.choice(indices)
            best_rand = min([d, e], key=lambda idx: self.fitness[idx])
            mutant = self.population[a] + self.mutation_factors[2] * (self.population[best_rand] - self.population[a])

        mutant = np.clip(mutant, self.bounds.lb, self.bounds.ub)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def adapt_strategy_probabilities(self, success, strategy, success_rate):
        self.success_history[strategy] = self.success_history[strategy] * self.fitness_decay_rate + success
        adjustment_rate = self.dynamic_adjustment_rate * (1 - (success_rate / (sum(self.success_history) + 1)))

        if success:
            self.strategy_probabilities = [
                min(p + adjustment_rate, 1.0) if idx == strategy else max(p - adjustment_rate, 0.0)
                for idx, p in enumerate(self.strategy_probabilities)
            ]
        else:
            self.strategy_probabilities = [
                max(p - adjustment_rate, 0.0) if idx == strategy else min(p + adjustment_rate, 1.0)
                for idx, p in enumerate(self.strategy_probabilities)
            ]
        
        total = sum(self.strategy_probabilities)
        self.strategy_probabilities = [p / total for p in self.strategy_probabilities]

        # Fitness-based adaptive mutation and crossover rate adjustments
        self.crossover_probability = self.crossover_probability * success_rate + (1 - success_rate) * self.dynamic_crossover_rate
        for i in range(len(self.mutation_factors)):
            self.mutation_factors[i] = self.mutation_factors[i] * success_rate + (1 - success_rate) * self.dynamic_mutation_adjustment

    def tournament_selection(self, func):
        tournament_size = 4 + np.random.randint(2)
        elite_idx = np.argmin(self.fitness)
        participants = np.random.choice(self.pop_size, tournament_size, replace=False)
        best_idx_among_participants = min(participants, key=lambda idx: self.fitness[idx])
        if np.random.rand() < self.elite_preservation_rate:
            return elite_idx
        else:
            return elite_idx if self.fitness[elite_idx] < self.fitness[best_idx_among_participants] else best_idx_among_participants

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population()
        self.evaluate_population(func)
        
        evaluations = self.pop_size
        while evaluations < self.budget:
            for _ in range(self.pop_size):
                i = self.tournament_selection(func)
                mutant = self.mutate(i)
                strategy = np.random.choice([0, 1, 2], p=self.strategy_probabilities)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1

                success_rate = 1.0 - (trial_fitness / (self.best_fitness + 1e-9))  # New: Compute success rate
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_individual = trial.copy()
                    self.adapt_strategy_probabilities(True, strategy, success_rate)
                else:
                    self.adapt_strategy_probabilities(False, strategy, success_rate)

                if evaluations >= self.budget:
                    break

        return self.best_individual, self.best_fitness