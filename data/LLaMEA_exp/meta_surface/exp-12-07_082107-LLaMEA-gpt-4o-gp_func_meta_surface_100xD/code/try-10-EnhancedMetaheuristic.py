import numpy as np

class EnhancedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = min(50, self.budget // 10)
        self.population_size = self.initial_population_size
        self.f = 0.5  # Initial differential weight
        self.cr = 0.9  # Initial crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.best_fitness = np.inf

    def chaotic_map(self, value):
        # Logistic map for chaotic behavior
        return 4.0 * value * (1.0 - value)

    def adapt_parameters(self, generation):
        # Use chaotic map for parameter adaptation
        self.f = max(0.1, self.chaotic_map(self.f))
        self.cr = min(1.0, self.chaotic_map(self.cr))
    
    def adjust_rates(self, trial_fitness, current_fitness):
        if trial_fitness < current_fitness:
            self.f = min(0.9, self.f + 0.1)  # Increase differential weight more to promote exploration
            self.cr = max(0.1, self.cr - 0.05)  # Decrease crossover probability slightly
    
    def adapt_population_size(self):
        # Reduce population size gradually as budget decreases
        if self.budget - self.evaluations < self.population_size:
            self.population_size = max(5, (self.budget - self.evaluations) // 2)

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

                # Mutation
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.f * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Evaluate
                trial_fitness = func(trial)
                self.evaluations += 1

                # Selection
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.adjust_rates(trial_fitness, self.fitness[i])

                # Update best fitness
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness

            generation += 1
        
        # Return the best found solution
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]