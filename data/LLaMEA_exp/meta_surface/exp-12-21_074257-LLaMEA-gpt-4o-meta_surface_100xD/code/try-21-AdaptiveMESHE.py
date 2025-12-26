import numpy as np

class AdaptiveMESHE:
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
        diversity_threshold = 0.1
        
        # Initialize population
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= population_size
        
        # Initialize memory with best individuals
        self.update_memory(population, fitness)
        
        while self.budget > 0:
            # Adaptive mutation rate based on diversity
            diversity = np.mean(np.std(population, axis=0))
            mutation_rate = max(initial_mutation_rate, diversity_threshold / (diversity + 1e-6))
            
            # Dynamic exploration based on crowding distance
            crowding_distances = self.calculate_crowding_distances(population)
            exploration_factor = np.max(crowding_distances) / (np.mean(crowding_distances) + 1e-6)
            
            offspring = []
            for rank, parent in enumerate(population):
                memory_sample = self.memory[np.random.choice(len(self.memory))]
                direction = memory_sample - parent
                adaptive_mutation_scale = (1 - (rank / population_size)) * mutation_rate
                if np.random.rand() < adaptive_mutation_scale * exploration_factor:
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

    def calculate_crowding_distances(self, population):
        distances = np.zeros(len(population))
        for i in range(self.dim):
            sorted_indices = np.argsort(population[:, i])
            sorted_population = population[sorted_indices]
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf
            for j in range(1, len(population) - 1):
                distances[sorted_indices[j]] += (sorted_population[j + 1, i] - sorted_population[j - 1, i])
        return distances