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
        self.strategy_memory = {'DE/rand/1/bin': 0, 'DE/best/1/bin': 0}  # Strategy success tracking

    def initialize_population(self, lb, ub):
        # Sine-Cosine initialization to ensure diverse initial population
        self.population = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            self.population[i] = lb + (ub - lb) * (r1 * np.sin(2 * np.pi * r2) + (1 - r1) * np.cos(2 * np.pi * r2))
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
        else:  # DE/best/1/bin
            best_idx = np.argmin(self.fitness)
            a, b = np.random.choice(indices, 2, replace=False)
            mutant = self.population[best_idx] + self.mutation_factor * (self.population[a] - self.population[b])
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
            self.strategy_memory[strategy] += 1

    def adapt_parameters(self):
        # Adaptive changes based on strategy success rate
        total_success = sum(self.strategy_memory.values())
        if total_success > 0:
            cr_weight = self.strategy_memory['DE/best/1/bin'] / total_success
            mf_weight = self.strategy_memory['DE/rand/1/bin'] / total_success
            self.crossover_rate = np.clip(0.5 + 0.3 * (cr_weight - mf_weight), 0.1, 0.9)
            self.mutation_factor = np.clip(0.8 + 0.2 * (mf_weight - cr_weight), 0.5, 1.0)
        else:
            self.crossover_rate = 0.5
            self.mutation_factor = 0.8

    def control_population_size(self):
        best_fitness = np.min(self.fitness)
        if self.dynamic_population_control and best_fitness < 1e-5:
            self.population_size = max(int(self.population_size * 0.9), self.dim * 2)

    def select_strategy(self):
        if np.random.rand() < 0.5:
            return 'DE/rand/1/bin'
        else:
            return 'DE/best/1/bin'
    
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

            # Apply local search to the best candidate
            best_idx = np.argmin(self.fitness)
            local_candidate, local_fitness = self.local_search(self.population[best_idx], func)
            if local_fitness < self.fitness[best_idx]:
                self.population[best_idx] = local_candidate
                self.fitness[best_idx] = local_fitness

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]