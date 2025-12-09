import numpy as np

class EnhancedDynamicMemoryMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = min(50, self.budget // 10)
        self.population_size = self.initial_population_size
        self.f = 0.5
        self.cr = 0.7
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.best_fitness = np.inf
        self.memory = []
        self.momentum = 0.1

    def chaotic_map(self, value):
        return 4.0 * value * (1.0 - value)

    def adapt_parameters(self, generation):
        self.f = max(0.1, self.chaotic_map(self.f))
        self.cr = min(1.0, self.chaotic_map(self.cr))

    def adjust_rates(self, trial_fitness, current_fitness):
        if trial_fitness < current_fitness:
            self.f = min(0.9, self.f + 0.05)
            self.cr = max(0.1, self.cr - 0.03)
        else:
            self.f = max(0.1, self.f - self.momentum * (trial_fitness - current_fitness) / (abs(current_fitness) + 1e-8))
            self.cr = min(0.9, self.cr + self.momentum * (trial_fitness - current_fitness) / (abs(current_fitness) + 1e-8))

    def adapt_population_size(self):
        if self.budget - self.evaluations < self.population_size:
            self.population_size = max(5, (self.budget - self.evaluations) // 2)

    def update_memory(self, trial, trial_fitness):
        self.memory.append((trial.copy(), trial_fitness))
        if len(self.memory) > self.population_size:
            self.memory.sort(key=lambda x: x[1])
            self.memory = self.memory[:self.population_size // 2] + self.memory[-self.population_size // 2:]

    def reintroduce_from_memory(self):
        if self.memory and np.random.rand() < 0.25:
            idx = np.random.choice(len(self.memory))
            return self.memory[idx][0]
        return None

    def __call__(self, func):
        generation = 0
        while self.evaluations < self.budget:
            self.adapt_population_size()
            self.population = self.population[:self.population_size]
            self.fitness = self.fitness[:self.population_size]
            
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                self.adapt_parameters(generation)

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.f * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                crossover_mask = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover_mask, mutant, self.population[i])

                memory_solution = self.reintroduce_from_memory()
                if memory_solution is not None:
                    trial = np.where(crossover_mask, memory_solution, trial)

                trial_fitness = func(trial)
                self.evaluations += 1

                self.update_memory(trial, trial_fitness)

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.adjust_rates(trial_fitness, self.fitness[i])

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness

            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            self.population[0] = elite

            generation += 1
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]