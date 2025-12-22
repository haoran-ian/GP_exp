import numpy as np

class EnhancedAdaptiveDETournamentV2:
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
        self.elite_preservation_rate = 0.1
        self.learning_rate = 0.01
        self.selection_pressure = 0.5  # New: initial selection pressure
        
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
            mutant = self.population[a] + self.mutation_factors[2] * (self.best_individual - self.population[a])

        mutant = np.clip(mutant, self.bounds.lb, self.bounds.ub)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def adapt_strategy_probabilities(self, success, strategy):
        self.success_history[strategy] += int(success)
        if success:
            self.strategy_probabilities[strategy] += self.learning_rate
        else:
            self.strategy_probabilities[strategy] -= self.learning_rate

        total = sum(self.strategy_probabilities)
        self.strategy_probabilities = [p / total for p in self.strategy_probabilities]

    def tournament_selection(self):
        tournament_size = int(np.ceil(self.selection_pressure * self.pop_size))
        participants = np.random.choice(self.pop_size, tournament_size, replace=False)
        best_idx = min(participants, key=lambda idx: self.fitness[idx])
        return best_idx

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population()
        self.evaluate_population(func)

        evaluations = self.pop_size
        while evaluations < self.budget:
            for _ in range(self.pop_size):
                i = self.tournament_selection()
                mutant = self.mutate(i)
                strategy = np.random.choice([0, 1, 2], p=self.strategy_probabilities)
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