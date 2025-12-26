import numpy as np

class EnhancedAMESH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = 10
        self.memory = []
        self.memory_decay_factor = 0.9
        self.adapt_factor = 0.99
        self.elite_preservation = 0.1  # New parameter for preserving best individuals

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10
        mutation_rate = 0.2
        sigma_init = 0.3
        diversity_threshold = 0.1
        
        # Initialize population
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= population_size

        # Initialize memory with best individuals and empty slots
        self.update_memory(population, fitness)
        
        while self.budget > 0:
            # Adaptive mutation rate based on diversity
            diversity = np.mean(np.std(population, axis=0))
            mutation_rate = max(mutation_rate * self.adapt_factor, diversity_threshold / (diversity + 1e-6))
            
            offspring = []
            num_elites = int(self.elite_preservation * population_size)
            elites = population[np.argsort(fitness)[:num_elites]]

            for rank, parent in enumerate(population):
                # Encourage exploration with memory and pure random direction
                if np.random.rand() < 0.5:
                    memory_sample = self.memory[np.random.choice(len(self.memory))]
                    direction = memory_sample - parent
                else:
                    direction = np.random.uniform(-1, 1, self.dim)
                
                # Multi-scale mutation
                adaptive_mutation_scale = (1 - (rank / population_size)) * mutation_rate
                scale_factors = np.random.uniform(0.5, 1.5, self.dim)  # Multi-scale factor
                direction += np.random.normal(0, sigma_init * adaptive_mutation_scale * scale_factors, self.dim)
                child = np.clip(parent + direction, lb, ub)
                offspring.append(child)

            # Evaluate offspring
            offspring_fitness = np.array([func(ind) for ind in offspring])
            self.budget -= population_size

            # Select the best individuals with elite preservation
            combined_population = np.vstack((population, offspring, elites))
            combined_fitness = np.hstack((fitness, offspring_fitness, fitness[:num_elites]))
            best_indices = np.argsort(combined_fitness)[:population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]
            
            # Update memory with best individuals
            self.update_memory(population, fitness)
        
        # Return the best solution found
        best_index = np.argmin(fitness)
        return population[best_index]
    
    def update_memory(self, population, fitness):
        best_individuals = population[np.argsort(fitness)[:self.memory_size]]
        self.memory = [(self.memory_decay_factor * old + (1 - self.memory_decay_factor) * new) for old, new in zip(self.memory, best_individuals)]
        if len(self.memory) < self.memory_size:
            self.memory.extend(best_individuals[:self.memory_size - len(self.memory)])
        else:
            self.memory = self.memory[:self.memory_size]