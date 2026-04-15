import numpy as np

class AdaptiveHybridExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_population_size = min(50, self.budget // 5)
        self.population_size = self.initial_population_size
        self.mutation_factor = 0.9
        self.crossover_rate = 0.9
        self.adaptation_threshold = 0.05
        self.dynamic_resize_factor = 1.1
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5

    def initialize_population(self, bounds):
        population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-(bounds.ub - bounds.lb), bounds.ub - bounds.lb, (self.population_size, self.dim))
        return population, velocities

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += len(population)
        return fitness

    def select_best(self, population, fitness):
        idx = np.argsort(fitness)
        return population[idx][:self.population_size // 2]

    def differential_evolution(self, population, bounds):
        offspring = []
        for i in range(len(population)):
            x_t = population[i]
            idxs = [idx for idx in range(len(population)) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.mutation_factor * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.crossover_rate
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            child = np.where(cross_points, mutant, x_t)
            offspring.append(child)
        return np.array(offspring)

    def update_velocities_and_positions(self, population, velocities, personal_best_positions, global_best_position, bounds):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        velocities = (self.inertia_weight * velocities +
                      self.cognitive_coefficient * r1 * (personal_best_positions - population) +
                      self.social_coefficient * r2 * (global_best_position - population))
        population = np.clip(population + velocities, bounds.lb, bounds.ub)
        return population, velocities

    def adapt_population_diversity(self, fitness):
        fitness_std = np.std(fitness)
        if fitness_std < self.adaptation_threshold:
            self.mutation_factor = min(self.mutation_factor * 1.2, 2.0)
            self.crossover_rate = min(self.crossover_rate * 1.1, 1.0)
            self.population_size = int(min(self.initial_population_size * self.dynamic_resize_factor, len(fitness) * self.dynamic_resize_factor))
        else:
            self.mutation_factor = max(self.mutation_factor * 0.8, 0.4)
            self.crossover_rate = max(self.crossover_rate * 0.9, 0.5)
            self.population_size = int(max(self.initial_population_size / self.dynamic_resize_factor, len(fitness) / self.dynamic_resize_factor))

    def __call__(self, func):
        bounds = func.bounds
        population, velocities = self.initialize_population(bounds)
        personal_best_positions = np.copy(population)
        personal_best_fitness = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_fitness = np.inf

        while self.evaluations < self.budget:
            fitness = self.evaluate_population(population, func)
            for i in range(self.population_size):
                if fitness[i] < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness[i]
                    personal_best_positions[i] = population[i]
            if np.min(fitness) < global_best_fitness:
                global_best_fitness = np.min(fitness)
                global_best_position = population[np.argmin(fitness)]
            
            self.adapt_population_diversity(fitness)
            parents = self.select_best(population, fitness)
            population = self.differential_evolution(parents, bounds)
            population, velocities = self.update_velocities_and_positions(population, velocities, personal_best_positions, global_best_position, bounds)
        
        return global_best_position, global_best_fitness