import numpy as np

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_rate = 0.1  # Initial mutation rate
        self.crossover_rate = 0.8  # Crossover probability
        self.elitism_rate = 0.1  # Percentage of elite individuals to retain
        self.adaptive_factor = 0.95  # Rate of adaptation for mutation and crossover

    def mutate(self, individual):
        mutation_vector = np.random.normal(0, 1, self.dim)
        return individual + self.mutation_rate * mutation_vector

    def crossover(self, parent1, parent2):
        mask = np.random.rand(self.dim) < self.crossover_rate
        return np.where(mask, parent1, parent2)

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_budget = self.population_size

        while eval_budget < self.budget:
            # Sort population by fitness
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]

            # Elitism: retain top performers
            elite_size = int(self.elitism_rate * self.population_size)
            elite_individuals = population[:elite_size]

            # Generate new offspring
            new_population = elite_individuals.tolist()
            while len(new_population) < self.population_size:
                parents = population[np.random.choice(elite_size, 2, replace=False)]
                offspring = self.crossover(parents[0], parents[1])
                offspring = self.mutate(offspring)
                offspring = np.clip(offspring, bounds[:, 0], bounds[:, 1])
                new_population.append(offspring)

            # Evaluate new population
            new_population = np.array(new_population)
            new_fitness = np.array([func(ind) for ind in new_population])
            eval_budget += self.population_size

            # Update population and fitness
            population = new_population
            fitness = new_fitness

            # Adaptive mutation and crossover adjustment
            diversity = np.std(fitness)
            if diversity < 1e-3:
                self.mutation_rate = max(0.01, self.mutation_rate * self.adaptive_factor)
                self.crossover_rate = min(0.9, self.crossover_rate / self.adaptive_factor)
            else:
                self.mutation_rate = min(0.5, self.mutation_rate / self.adaptive_factor)
                self.crossover_rate = max(0.6, self.crossover_rate * self.adaptive_factor)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]