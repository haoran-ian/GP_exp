import numpy as np

class HybridDELevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = None
        self.fitness = None

    def levy_flight(self, step_size=0.01):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / (np.abs(v) ** (1 / beta))
        # Adapt step size based on population diversity
        diversity = np.std(self.population, axis=0).mean()
        return (step_size + 0.05 * diversity) * step

    def initialize_population(self):
        self.population = np.random.uniform(self.lower_bound, self.upper_bound,
                                            (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
    
    def mutation(self, idx, best_idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        improvement_factor = 1 - (self.fitness[idx] - self.fitness[best_idx]) / abs(self.fitness[idx] - self.fitness[best_idx] + 1e-8)
        dynamic_mutation_factor = self.mutation_factor * improvement_factor
        mutated = (self.population[a] + 
                   dynamic_mutation_factor * (self.population[b] - self.population[c] +
                                              self.population[best_idx] - self.population[idx]))
        mutated = np.clip(mutated, self.lower_bound, self.upper_bound)
        return mutated
    
    def crossover(self, target, mutant):
        crossover_vector = np.copy(target)
        rand_idx = np.random.randint(self.dim)
        adaptive_crossover_prob = self.crossover_prob * (1 - self.budget_used / self.budget) + 0.1
        for j in range(self.dim):
            if np.random.rand() < adaptive_crossover_prob or j == rand_idx:
                crossover_vector[j] = mutant[j]
        return crossover_vector

    def select_best(self):
        return np.argmin(self.fitness)

    def __call__(self, func):
        self.initialize_population()
        func_evals = 0
        self.budget_used = 0
        while func_evals < self.budget:
            self.evaluate_population(func)
            best_idx = self.select_best()
            global_best = self.population[best_idx]
            for i in range(self.population_size):
                if func_evals >= self.budget:
                    break
                mutant = self.mutation(i, best_idx)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                func_evals += 1
                self.budget_used = func_evals
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                else:
                    levy_step = self.levy_flight() + 0.1 * global_best
                    levy_trial = np.clip(self.population[i] + levy_step, 
                                         self.lower_bound, self.upper_bound)
                    levy_fitness = func(levy_trial)
                    func_evals += 1
                    self.budget_used = func_evals
                    if levy_fitness < self.fitness[i]:
                        self.population[i] = levy_trial
                        self.fitness[i] = levy_fitness
        return self.population[self.select_best()]