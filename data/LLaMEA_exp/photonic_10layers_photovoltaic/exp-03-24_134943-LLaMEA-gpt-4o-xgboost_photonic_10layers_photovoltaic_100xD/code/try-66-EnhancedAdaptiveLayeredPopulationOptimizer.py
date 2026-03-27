import numpy as np

class EnhancedAdaptiveLayeredPopulationOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.base_population_size = 30
        self.inner_population_size = 15
        self.populations = 3
        self.phase_detection_threshold = 1e-3
        self.exploration_factors = np.linspace(0.9, 1.1, self.populations)
        self.exploitation_factors = np.linspace(0.2, 0.3, self.populations)
        self.dynamic_ratios = np.linspace(0.05, 0.15, self.populations)
        self.learning_rates = np.linspace(0.01, 0.1, self.populations)
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1

    def initialize_populations(self, bounds):
        return [np.random.uniform(bounds.lb, bounds.ub, (self.base_population_size, self.dim)) for _ in range(self.populations)]

    def evaluate_population(self, population, func):
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += len(population)
        return fitness

    def detect_phase_transition(self, population):
        distances = np.linalg.norm(np.diff(population, axis=0), axis=1)
        return np.mean(distances) > self.phase_detection_threshold

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dim - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
        else:
            child1, child2 = parent1, parent2
        return child1, child2

    def mutate(self, individual, bounds):
        if np.random.rand() < self.mutation_rate:
            mutation_vector = np.random.uniform(bounds.lb, bounds.ub, self.dim)
            return np.clip(individual + mutation_vector * 0.1, bounds.lb, bounds.ub)
        return individual

    def exploit(self, best_solution, bounds):
        noise = np.random.uniform(-0.5, 0.5, (self.inner_population_size, self.dim))
        return np.clip(best_solution + self.exploitation_factors * noise, bounds.lb, bounds.ub)

    def explore(self, bounds):
        base_exploration = np.random.uniform(bounds.lb, bounds.ub, (self.inner_population_size, self.dim))
        return base_exploration

    def context_aware_transition(self, populations, fitnesses):
        for i in range(self.populations):
            if self.detect_phase_transition(populations[i]):
                self.exploitation_factors[i] = min(self.exploitation_factors[i] * 1.5, 1.0)
            else:
                self.exploration_factors[i] = max(self.exploration_factors[i] * 0.9, 0.5)

    def __call__(self, func):
        bounds = func.bounds
        populations = self.initialize_populations(bounds)
        fitnesses = [self.evaluate_population(population, func) for population in populations]

        while self.evaluations < self.budget:
            self.context_aware_transition(populations, fitnesses)
            new_populations, new_fitnesses = [], []

            for i in range(self.populations):
                best_idx = np.argmin(fitnesses[i])
                best_solution = populations[i][best_idx]

                if np.random.rand() < 0.6:
                    new_population = self.exploit(best_solution, bounds)
                else:
                    new_population = self.explore(bounds)

                # Apply crossover and mutation
                for j in range(0, self.inner_population_size, 2):
                    parent1, parent2 = new_population[j], new_population[min(j+1, self.inner_population_size-1)]
                    child1, child2 = self.crossover(parent1, parent2)
                    new_population[j] = self.mutate(child1, bounds)
                    new_population[min(j+1, self.inner_population_size-1)] = self.mutate(child2, bounds)

                new_fitness = self.evaluate_population(new_population, func)
                combined_population = np.vstack((populations[i], new_population))
                combined_fitness = np.concatenate((fitnesses[i], new_fitness))
                best_indices = np.argsort(combined_fitness)[:self.base_population_size]
                new_populations.append(combined_population[best_indices])
                new_fitnesses.append(combined_fitness[best_indices])

            populations, fitnesses = new_populations, new_fitnesses

        best_idx = np.argmin([np.min(fitness) for fitness in fitnesses])
        return populations[best_idx][np.argmin(fitnesses[best_idx])]