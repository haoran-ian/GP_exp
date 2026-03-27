import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.local_search_prob = 0.3
        self.pop = []
        self.best_solution = None
        self.bounds = None
        self.fitness_history = []

    def initialize_population(self):
        self.pop = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.pop])
        return fitness

    def mutation(self, idx):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant_vector = np.clip(self.pop[a] + self.mutation_factor * (self.pop[b] - self.pop[c]), self.bounds.lb, self.bounds.ub)
        return mutant_vector

    def crossover(self, target_vector, mutant_vector):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial_vector = np.where(crossover_mask, mutant_vector, target_vector)
        return trial_vector

    def adaptive_local_search(self, vector, func):
        new_vector = np.copy(vector)
        for _ in range(self.dim):
            if np.random.rand() < self.local_search_prob:
                perturbation = np.random.normal(0, 1, self.dim) * (self.bounds.ub - self.bounds.lb) / 10
                new_vector = np.clip(new_vector + perturbation, self.bounds.lb, self.bounds.ub)
        return new_vector if func(new_vector) < func(vector) else vector

    def adjust_parameters(self, iteration):
        # Dynamically adjust mutation factor and crossover rate based on iteration
        self.mutation_factor = 0.5 + 0.3 * (1 - iteration / self.budget)
        self.crossover_rate = 0.9 - 0.5 * (iteration / self.budget)
        self.local_search_prob = 0.3 + 0.2 * (iteration / self.budget)

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population()
        fitness = self.evaluate_population(func)
        self.best_solution = self.pop[np.argmin(fitness)]
        self.fitness_history.append(np.min(fitness))

        for iteration in range(self.budget - self.population_size):
            self.adjust_parameters(iteration)
            for i in range(self.population_size):
                mutant_vector = self.mutation(i)
                trial_vector = self.crossover(self.pop[i], mutant_vector)
                trial_vector = self.adaptive_local_search(trial_vector, func)
                trial_fitness = func(trial_vector)

                if trial_fitness < fitness[i]:
                    self.pop[i] = trial_vector
                    fitness[i] = trial_fitness

                if trial_fitness < func(self.best_solution):
                    self.best_solution = trial_vector

            self.fitness_history.append(np.min(fitness))
        
        return self.best_solution