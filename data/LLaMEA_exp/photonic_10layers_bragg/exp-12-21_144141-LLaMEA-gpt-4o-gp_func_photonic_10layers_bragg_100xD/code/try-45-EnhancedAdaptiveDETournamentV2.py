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
        self.dynamic_adjustment_rate = 0.03
        self.elite_preservation_rate = 0.2
        self.tournament_pressure = 0.75  # New: Probability of selecting more fit individuals

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
        adjustment = self.dynamic_adjustment_rate * (1 if success else -1)
        self.strategy_probabilities[strategy] += adjustment
        self.strategy_probabilities = np.clip(self.strategy_probabilities, 0.1, 0.8)
        total = sum(self.strategy_probabilities)
        self.strategy_probabilities = [p / total for p in self.strategy_probabilities]

    def tournament_selection(self):
        tournament_size = 4 + np.random.randint(2)
        participants = np.random.choice(self.pop_size, tournament_size, replace=False)
        fitness_sorted_idxs = sorted(participants, key=lambda idx: self.fitness[idx])

        prob = np.random.uniform()
        for i, idx in enumerate(fitness_sorted_idxs):
            if prob < self.tournament_pressure * (1 - (i / tournament_size)):
                return idx
        return fitness_sorted_idxs[0]

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population()
        self.evaluate_population(func)

        evaluations = self.pop_size
        while evaluations < self.budget:
            for _ in range(self.pop_size):
                i = self.tournament_selection()
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_individual = trial.copy()
                    self.adapt_strategy_probabilities(True, np.random.choice([0, 1, 2], p=self.strategy_probabilities))
                else:
                    self.adapt_strategy_probabilities(False, np.random.choice([0, 1, 2], p=self.strategy_probabilities))

                if evaluations >= self.budget:
                    break

        return self.best_individual, self.best_fitness