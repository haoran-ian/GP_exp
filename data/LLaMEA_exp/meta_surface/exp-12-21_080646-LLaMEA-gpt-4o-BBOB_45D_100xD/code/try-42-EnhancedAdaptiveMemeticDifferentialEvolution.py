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
        self.strategy_memory = {'DE/rand/1/bin': 0, 'DE/best/1/bin': 0, 'DE/current-to-best/1/bin': 0}

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.fitness = np.full(self.population_size, np.inf)

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
        else:  # DE/current-to-best/1/bin
            best_idx = np.argmin(self.fitness)
            a, b = np.random.choice(indices, 2, replace=False)
            mutant = self.population[target_idx] + self.mutation_factor * (self.population[best_idx] - self.population[target_idx]) + self.mutation_factor * (self.population[a] - self.population[b])
        
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

    def adapt_parameters(self):
        self.crossover_rate = max(0.1, min(0.9, self.crossover_rate + np.random.normal(0, 0.1)))
        self.mutation_factor = max(0.5, min(1.0, self.mutation_factor + np.random.normal(0, 0.1)))

    def control_population_size(self):
        best_fitness = np.min(self.fitness)
        if self.dynamic_population_control and best_fitness < 1e-5:
            self.population_size = max(int(self.population_size * 0.9), self.dim * 2)
            if self.population_size < len(self.population):
                self.population = self.population[:self.population_size]
                self.fitness = self.fitness[:self.population_size]

    def select_strategy(self):
        total = sum(self.strategy_memory.values())
        if total == 0:
            return 'DE/rand/1/bin'
        probabilities = {k: v / total for k, v in self.strategy_memory.items()}
        return np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))
    
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

            best_idx = np.argmin(self.fitness)
            local_candidate, local_fitness = self.local_search(self.population[best_idx], func)
            if local_fitness < self.fitness[best_idx]:
                self.population[best_idx] = local_candidate
                self.fitness[best_idx] = local_fitness

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]