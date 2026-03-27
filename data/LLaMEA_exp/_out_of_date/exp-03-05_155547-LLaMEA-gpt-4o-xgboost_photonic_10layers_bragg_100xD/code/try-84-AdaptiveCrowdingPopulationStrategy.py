import numpy as np

class AdaptiveCrowdingPopulationStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)
        self.cr = 0.9  # Crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = self.random_state.uniform(lb, ub, size=(population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.budget:
            elite_count = int(self.elite_fraction * population_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite = population[elite_indices]

            offspring = []
            for _ in range(population_size - elite_count):
                parent1 = self.select_by_crowding_distance(elite, fitness[elite_indices])
                parent2 = self.select_by_crowding_distance(elite, fitness[elite_indices])
                child = self.crossover(parent1, parent2, lb, ub, np.std(fitness))
                offspring.append(self.mutate(child, lb, ub, np.std(fitness)))

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))
            
            # Dynamically adjust population size
            population_size = max(10, int(self.initial_population_size * (1 - evaluations/self.budget)))

        best_index = np.argmin(fitness)
        return population[best_index]

    def select_by_crowding_distance(self, population, fitness):
        if len(population) < 2:
            return population[0]
        
        distances = np.zeros(len(population))
        sorted_indices = np.argsort(fitness)
        for i in range(1, len(population) - 1):
            distances[sorted_indices[i]] = (
                np.linalg.norm(population[sorted_indices[i + 1]] - population[sorted_indices[i - 1]])
            )
        distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf
        selected_index = np.argmax(distances)
        return population[selected_index]

    def crossover(self, parent1, parent2, lb, ub, diversity):
        self.cr = 0.7 + 0.2 * diversity  # Adjust crossover probability based on diversity
        mask = self.random_state.rand(self.dim) < self.cr
        child = np.where(mask, parent1, parent2)
        return np.clip(child, lb, ub)

    def mutate(self, individual, lb, ub, diversity):
        mutation_strength = self.random_state.rand() * 0.1 * diversity  # Adjust mutation strength based on diversity
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = individual + noise
        return np.clip(mutant, lb, ub)