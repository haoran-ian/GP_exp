import numpy as np

class MultiStrategyAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.population = None
        self.fitness = None
        self.memory_archive = []
        self.strategy_probabilities = [0.5, 0.5]  # Initial probability for each strategy

    def initialize(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def mutate_current_to_best(self, idx, best_idx):
        idxs = [i for i in range(self.population_size) if i != idx and i != best_idx]
        a, b = np.random.choice(idxs, 2, replace=False)
        F_best = 0.5  # Dynamic differential weight
        mutant = self.population[idx] + F_best * (self.population[best_idx] - self.population[idx]) + F_best * (self.population[a] - self.population[b])
        return mutant

    def mutate_rand_1(self, idx):
        idxs = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        F_rand = 0.8  # Static differential weight
        mutant = self.population[a] + F_rand * (self.population[b] - self.population[c])
        return mutant

    def crossover(self, target, mutant):
        CR = 0.9  # Static crossover probability
        crossover_mask = np.random.rand(self.dim) < CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def adaptive_local_search(self, candidate):
        perturbation_strength = 0.1 * (np.random.rand(self.dim) - 0.5)
        return candidate + perturbation_strength

    def update_strategy_probabilities(self, improvement, strategy_index):
        learning_rate = 0.1
        decay = 0.99
        for i in range(len(self.strategy_probabilities)):
            if i == strategy_index:
                self.strategy_probabilities[i] += learning_rate * improvement
            self.strategy_probabilities[i] *= decay
        total = sum(self.strategy_probabilities)
        self.strategy_probabilities = [p / total for p in self.strategy_probabilities]

    def update_memory_archive(self, candidate, candidate_fitness):
        self.memory_archive.append((candidate, candidate_fitness))
        if len(self.memory_archive) > self.population_size:
            self.memory_archive.pop(0)

    def __call__(self, func):
        self.initialize(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            best_idx = np.argmin(self.fitness)

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                strategy_index = np.random.choice(len(self.strategy_probabilities), p=self.strategy_probabilities)
                
                if strategy_index == 0:
                    mutant = self.mutate_current_to_best(i, best_idx)
                else:
                    mutant = self.mutate_rand_1(i)
                
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
                
                trial_fitness = func(trial)
                evaluations += 1
                
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    improvement = self.fitness[i] - trial_fitness
                    self.fitness[i] = trial_fitness
                    self.update_memory_archive(trial, trial_fitness)
                else:
                    improvement = 0
                
                # Apply local search with a memory-based adaptive probability
                if np.random.rand() < min(0.3, 0.5 * (1 - evaluations / self.budget)):
                    candidate = self.adaptive_local_search(self.population[i])
                    candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)
                    
                    candidate_fitness = func(candidate)
                    evaluations += 1
                    
                    if candidate_fitness < self.fitness[i]:
                        self.population[i] = candidate
                        self.fitness[i] = candidate_fitness
                        self.update_memory_archive(candidate, candidate_fitness)

                self.update_strategy_probabilities(improvement, strategy_index)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]