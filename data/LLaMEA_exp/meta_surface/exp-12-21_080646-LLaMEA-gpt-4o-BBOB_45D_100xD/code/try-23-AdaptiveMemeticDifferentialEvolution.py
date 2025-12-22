import numpy as np

class AdaptiveMemeticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.population = None
        self.bounds = None
        self.fitness = None
        self.crossover_rate = 0.7  # Adjusted for better exploration-exploitation balance
        self.mutation_factor = 0.9  # Adjusted for larger mutation steps
        self.generations = 0
        self.dynamic_population_control = True
        self.memory = []
        self.strategy_memory = {'DE/rand/1/bin': 0, 'DE/best/1/bin': 0, 'DE/rand/2/bin': 0}  # Added new strategy

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
        elif strategy == 'DE/rand/2/bin':  # New mutation strategy
            a, b, c, d, e = np.random.choice(indices, 5, replace=False)
            mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c] + self.population[d] - self.population[e])
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
            self.memory.append((self.crossover_rate, self.mutation_factor))
            self.strategy_memory[strategy] += 1

    def adapt_parameters(self):
        if self.memory:
            self.crossover_rate, self.mutation_factor = self._calculate_adaptive_parameters(self.memory)
        else:
            self.crossover_rate = 0.2 + np.random.rand() * 0.8
            self.mutation_factor = 0.7 + np.random.rand() * 0.3

    def _calculate_adaptive_parameters(self, memory):
        average_cr = np.clip(np.mean([cr for cr, _ in memory]) + np.random.normal(0, 0.05), 0.1, 0.9)
        average_f = np.clip(np.mean([mf for _, mf in memory]) + np.random.normal(0, 0.05), 0.5, 1.0)
        return average_cr, average_f

    def control_population_size(self):
        best_fitness = np.min(self.fitness)
        if self.dynamic_population_control and best_fitness < 1e-5:
            self.population_size = max(int(self.population_size * 0.85), self.dim * 2)  # Adjusted shrink factor

    def select_strategy(self):
        weights = [self.strategy_memory[key] for key in self.strategy_memory]
        weights = np.array(weights) / np.sum(weights)
        return np.random.choice(list(self.strategy_memory.keys()), p=weights)
    
    def local_search(self, candidate, func):
        perturbation = (np.random.randn(self.dim) - 0.5) * 0.05 * (self.bounds.ub - self.bounds.lb)  # Adjusted for smaller steps
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