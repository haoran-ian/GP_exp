import numpy as np

class RefinedMetaheuristic:
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
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.global_best = None
        self.global_best_fitness = np.inf

    def chaotic_map(self, value):
        return 4.0 * value * (1.0 - value)

    def adapt_parameters(self, generation):
        self.f = max(0.1, self.chaotic_map(self.f))
        self.cr = min(1.0, self.chaotic_map(self.cr))
    
    def adjust_rates(self, trial_fitness, current_fitness):
        if trial_fitness < current_fitness:
            self.f = min(0.9, self.f + 0.05)
            self.cr = max(0.1, self.cr - 0.05)
    
    def adapt_population_size(self):
        if self.budget - self.evaluations < self.population_size:
            self.population_size = max(5, (self.budget - self.evaluations) // 2)

    def __call__(self, func):
        generation = 0
        while self.evaluations < self.budget:
            self.adapt_population_size()
            self.population = self.population[:self.population_size]
            self.fitness = self.fitness[:self.population_size]
            self.velocity = self.velocity[:self.population_size]
            self.personal_best = self.personal_best[:self.population_size]
            self.personal_best_fitness = self.personal_best_fitness[:self.population_size]

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                self.adapt_parameters(generation)

                # Particle Swarm Update
                inertia = 0.7
                cognitive_coeff = 1.5
                social_coeff = 1.5
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocity[i] = (inertia * self.velocity[i] +
                                    cognitive_coeff * r1 * (self.personal_best[i] - self.population[i]) +
                                    social_coeff * r2 * (self.global_best - self.population[i]))
                self.velocity[i] = np.clip(self.velocity[i], -1, 1)
                candidate = self.population[i] + self.velocity[i]
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)

                # Differential Evolution Mutation and Crossover
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.f * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover_mask, mutant, candidate)

                # Evaluate
                trial_fitness = func(trial)
                self.evaluations += 1

                # Selection and Personal Best Update
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.adjust_rates(trial_fitness, self.fitness[i])
                if trial_fitness < self.personal_best_fitness[i]:
                    self.personal_best[i] = trial
                    self.personal_best_fitness[i] = trial_fitness

                # Global Best Update
                if trial_fitness < self.global_best_fitness:
                    self.global_best = trial
                    self.global_best_fitness = trial_fitness

            generation += 1
        
        return self.global_best, self.global_best_fitness