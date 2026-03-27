import numpy as np

class EnhancedDifferentialEvolutionV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 15 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.local_search_prob = 0.3
        self.pop = []
        self.best_solution = None
        self.bounds = None

    def initialize_population(self):
        self.pop = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.pop])
        return fitness

    def update_mutation_factor(self, fitness):
        avg_fitness = np.mean(fitness)
        self.mutation_factor = 0.5 + 0.5 * np.random.rand() * (fitness - avg_fitness) / (np.max(fitness) - np.min(fitness) + 1e-8)

    def update_crossover_rate(self, gen):
        max_gen = self.budget // self.population_size
        self.crossover_rate = 0.9 - 0.5 * (gen / max_gen)

    def mutation(self, idx, fitness):
        self.update_mutation_factor(fitness)
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant_vector = np.clip(self.pop[a] + self.mutation_factor * (self.pop[b] - self.pop[c]), self.bounds.lb, self.bounds.ub)
        return mutant_vector

    def crossover(self, target_vector, mutant_vector):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial_vector = np.where(crossover_mask, mutant_vector, target_vector)
        return trial_vector

    def localized_adaptive_search(self, vector, func):
        new_vector = np.copy(vector)
        perturbation_scale = np.random.uniform(0.1, 0.5) * (self.bounds.ub - self.bounds.lb)
        for _ in range(self.dim):
            if np.random.rand() < self.local_search_prob:
                perturbation = np.random.normal(0, 1, self.dim) * perturbation_scale
                candidate_vector = np.clip(new_vector + perturbation, self.bounds.lb, self.bounds.ub)
                if func(candidate_vector) < func(new_vector):
                    new_vector = candidate_vector
        return new_vector

    def maintain_diversity(self):
        for i in range(self.population_size):
            if np.random.rand() < 0.1:  # Randomly regenerate some individuals
                self.pop[i] = np.random.uniform(self.bounds.lb, self.bounds.ub, self.dim)

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population()
        fitness = self.evaluate_population(func)
        self.best_solution = self.pop[np.argmin(fitness)]

        for gen in range(self.budget - self.population_size):
            self.update_crossover_rate(gen)
            for i in range(self.population_size):
                mutant_vector = self.mutation(i, fitness)
                trial_vector = self.crossover(self.pop[i], mutant_vector)
                trial_vector = self.localized_adaptive_search(trial_vector, func)
                trial_fitness = func(trial_vector)
                
                if trial_fitness < fitness[i]:
                    self.pop[i] = trial_vector
                    fitness[i] = trial_fitness
                
                if trial_fitness < func(self.best_solution):
                    self.best_solution = trial_vector
            
            self.maintain_diversity()
        
        return self.best_solution