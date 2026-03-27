import numpy as np

class EnhancedMultiPopulationPhaseTransitionOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_population_size = 30
        self.populations = 3
        self.phase_detection_threshold = 1e-3
        self.exploration_factors = np.linspace(0.9, 1.1, self.populations)
        self.exploitation_factors = np.linspace(0.2, 0.3, self.populations)
        self.dynamic_ratios = np.linspace(0.05, 0.15, self.populations)
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7

    def initialize_populations(self, bounds):
        return [np.random.uniform(bounds.lb, bounds.ub, (self.initial_population_size, self.dim)) for _ in range(self.populations)]

    def evaluate_population(self, population, func):
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += len(population)
        return fitness

    def detect_phase_transition(self, population):
        distances = np.linalg.norm(np.diff(population, axis=0), axis=1)
        return np.mean(distances) > self.phase_detection_threshold

    def exploit(self, best_solution, bounds, exploitation_factor):
        noise = np.random.uniform(-0.5, 0.5, (self.initial_population_size, self.dim))
        return np.clip(best_solution + exploitation_factor * noise, bounds.lb, bounds.ub)

    def explore(self, bounds, exploration_factor):
        return np.random.uniform(bounds.lb, bounds.ub, (self.initial_population_size, self.dim)) + \
               np.random.normal(0, exploration_factor, (self.initial_population_size, self.dim))

    def dynamically_adjust_population(self, population_size, dynamic_ratio):
        return max(int(population_size * (1 + dynamic_ratio * (self.evaluations / self.budget))), 1)

    def refine_local_search_space(self, best_solution, bounds, radius=0.1):
        lower_bound = np.maximum(bounds.lb, best_solution - radius * (bounds.ub - bounds.lb))
        upper_bound = np.minimum(bounds.ub, best_solution + radius * (bounds.ub - bounds.lb))
        return np.random.uniform(lower_bound, upper_bound, (self.initial_population_size, self.dim))

    def mutate(self, population, bounds):
        mutation_mask = np.random.rand(*population.shape) < self.mutation_rate
        potential_mutations = np.random.uniform(bounds.lb, bounds.ub, population.shape)
        return np.where(mutation_mask, potential_mutations, population)

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.dim)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        return parent1, parent2

    def __call__(self, func):
        bounds = func.bounds
        populations = self.initialize_populations(bounds)
        fitnesses = [self.evaluate_population(population, func) for population in populations]
        population_sizes = [self.initial_population_size for _ in range(self.populations)]

        while self.evaluations < self.budget:
            new_populations, new_fitnesses = [], []

            for i in range(self.populations):
                population_sizes[i] = self.dynamically_adjust_population(population_sizes[i], self.dynamic_ratios[i])
                best_idx = np.argmin(fitnesses[i])
                best_solution = populations[i][best_idx]

                if np.random.rand() < 0.6 and self.detect_phase_transition(populations[i]):
                    new_population = self.explore(bounds, self.exploration_factors[i])
                else:
                    new_population = self.refine_local_search_space(best_solution, bounds)

                # Apply mutation
                new_population = self.mutate(new_population, bounds)

                # Apply crossover
                new_population_size = new_population.shape[0]
                for j in range(0, new_population_size, 2):
                    if j + 1 < new_population_size:
                        new_population[j], new_population[j+1] = self.crossover(new_population[j], new_population[j+1])

                new_fitness = self.evaluate_population(new_population, func)

                combined_population = np.vstack((populations[i], new_population))
                combined_fitness = np.concatenate((fitnesses[i], new_fitness))
                best_indices = np.argsort(combined_fitness)[:population_sizes[i]]
                new_populations.append(combined_population[best_indices])
                new_fitnesses.append(combined_fitness[best_indices])

            populations, fitnesses = new_populations, new_fitnesses

        best_idx = np.argmin([np.min(fitness) for fitness in fitnesses])
        return populations[best_idx][np.argmin(fitnesses[best_idx])]