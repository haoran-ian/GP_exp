import numpy as np

class AdaptiveDEWithReinforcement:
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
        self.mutation_factors = [0.5, 0.8]  # Mutation factors for the strategies
        self.crossover_probability = 0.9
        self.reward = [0.0, 0.0]  # Reward scores for strategies

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
        return mutant, strategy

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def adapt_strategy_probabilities(self):
        total_reward = sum(self.reward)
        if total_reward > 0:
            self.strategy_probabilities = [r / total_reward for r in self.reward]
        else:
            self.strategy_probabilities = [0.5, 0.5]  # Reset to equal probability

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population()
        self.evaluate_population(func)
        
        evaluations = self.pop_size
        while evaluations < self.budget:
            for i in range(self.pop_size):
                mutant, strategy = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_individual = trial.copy()
                    self.reward[strategy] += 1.0  # Reward the successful strategy
                else:
                    self.reward[strategy] -= 0.5  # Penalize the unsuccessful strategy

                self.adapt_strategy_probabilities()

                if evaluations >= self.budget:
                    break

        return self.best_individual, self.best_fitness