import numpy as np

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.adaptive_local_search_prob = 0.3
        self.pop = []
        self.best_solution = None
        self.bounds = None
        self.fitness = None
        self.dynamic_scaling_factor = 0.9

    def initialize_population(self):
        self.pop = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i, ind in enumerate(self.pop):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(ind)

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
        perturbation_scale = self.dynamic_scaling_factor * (self.bounds.ub - self.bounds.lb) / 10
        for _ in range(self.dim):
            if np.random.rand() < self.adaptive_local_search_prob:
                perturbation = np.random.normal(0, 1, self.dim) * perturbation_scale
                candidate_vector = np.clip(new_vector + perturbation, self.bounds.lb, self.bounds.ub)
                if func(candidate_vector) < func(new_vector):
                    new_vector = candidate_vector
        return new_vector

    def dynamic_population_resizing(self, iteration):
        scale_down_factor = 1 - (iteration / (2 * self.budget))
        self.population_size = max(4, int(self.population_size * scale_down_factor))
        self.pop = self.pop[:self.population_size]
        self.fitness = self.fitness[:self.population_size]

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population()
        self.evaluate_population(func)
        self.best_solution = self.pop[np.argmin(self.fitness)]
        evaluations = self.population_size
        
        iteration = 0
        while evaluations < self.budget:
            self.dynamic_population_resizing(iteration)
            for i in range(self.population_size):
                mutant_vector = self.mutation(i)
                trial_vector = self.crossover(self.pop[i], mutant_vector)
                trial_vector = self.adaptive_local_search(trial_vector, func)
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.pop[i] = trial_vector
                    self.fitness[i] = trial_fitness
                
                if trial_fitness < func(self.best_solution):
                    self.best_solution = trial_vector
            
            iteration += 1

        return self.best_solution