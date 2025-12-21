import numpy as np

class AdvancedAMESH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = 5
        self.memory = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10
        initial_mutation_rate = 0.2
        sigma_init = 0.3
        crossover_rate = 0.8

        # Initialize population
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= population_size

        # Initialize memory with best individuals
        self.update_memory(population, fitness)

        while self.budget > 0:
            # Adaptive mutation rate based on diversity
            diversity = np.mean(np.std(population, axis=0))
            mutation_rate = max(initial_mutation_rate, 0.1 / (diversity + 1e-6))

            offspring = []
            for rank, parent in enumerate(population):
                if np.random.rand() < crossover_rate:
                    partner = population[np.random.choice(len(population))]
                    child = self.crossover(parent, partner, lb, ub)
                else:
                    memory_sample = self.memory[np.random.choice(len(self.memory))]
                    direction = memory_sample - parent
                    adaptive_mutation_scale = (1 - (rank / population_size)) * mutation_rate
                    if np.random.rand() < adaptive_mutation_scale:
                        direction += np.random.normal(0, sigma_init, self.dim)
                    child = np.clip(parent + direction, lb, ub)
                offspring.append(child)

            # Evaluate offspring
            offspring_fitness = np.array([func(ind) for ind in offspring])
            self.budget -= population_size

            # Select the best individuals
            combined_population = np.vstack((population, offspring))
            combined_fitness = np.hstack((fitness, offspring_fitness))
            best_indices = np.argsort(combined_fitness)[:population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

            # Update memory with best individuals and maintain diversity
            self.update_memory(population, fitness)

        # Return the best solution found
        best_index = np.argmin(fitness)
        return population[best_index]

    def update_memory(self, population, fitness):
        best_individuals = population[np.argsort(fitness)[:self.memory_size]]
        self.memory = list(best_individuals)

    def crossover(self, parent1, parent2, lb, ub):
        alpha = np.random.uniform(0, 1, self.dim)
        child = alpha * parent1 + (1 - alpha) * parent2
        return np.clip(child, lb, ub)