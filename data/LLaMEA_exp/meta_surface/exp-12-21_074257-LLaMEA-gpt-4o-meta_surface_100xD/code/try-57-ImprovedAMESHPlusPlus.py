import numpy as np

class ImprovedAMESHPlusPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = 15  # Increased memory size for more diversity
        self.memory = []
        self.memory_decay_factor = 0.85  # Slightly reduced decay for better memory retention
        self.adapt_factor = 0.98  # Adjusted adaptation parameter for learning rate
        self.learn_rate = 0.05  # New parameter for adaptive learning rate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 15  # Increased population size for greater exploration
        mutation_rate = 0.25
        sigma_init = 0.3
        diversity_threshold = 0.05  # More sensitive diversity threshold
        
        # Initialize population
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= population_size

        # Initialize memory with best individuals and empty slots
        self.update_memory(population, fitness)
        
        while self.budget > 0:
            # Adaptive mutation rate based on diversity
            diversity = np.mean(np.std(population, axis=0))
            mutation_rate = max(mutation_rate * (self.adapt_factor + self.learn_rate * diversity), 
                                diversity_threshold / (diversity + 1e-6))
            
            offspring = []
            for rank, parent in enumerate(population):
                if np.random.rand() < 0.5 or not self.memory:  # Exploration with memory or random direction
                    if self.memory:
                        memory_sample = self.memory[np.random.choice(len(self.memory))]
                        direction = memory_sample - parent
                    else:
                        direction = np.random.uniform(-1, 1, self.dim)
                else:  # Pure exploration
                    direction = np.random.uniform(-1, 1, self.dim)
                
                adaptive_mutation_scale = (1 - (rank / population_size)) * mutation_rate
                direction += np.random.normal(0, sigma_init * adaptive_mutation_scale, self.dim)
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
            
            # Update memory with best individuals and diversify
            self.update_memory(population, fitness)
        
        # Return the best solution found
        best_index = np.argmin(fitness)
        return population[best_index]
    
    def update_memory(self, population, fitness):
        best_individuals = population[np.argsort(fitness)[:self.memory_size]]
        # Apply memory decay factor to reinforce new best individuals
        self.memory = [(self.memory_decay_factor * old + (1 - self.memory_decay_factor) * new) 
                       for old, new in zip(self.memory, best_individuals)]
        # Fill the memory with new individuals if size is not met
        if len(self.memory) < self.memory_size:
            self.memory.extend(best_individuals[:self.memory_size - len(self.memory)])
        else:
            self.memory = self.memory[:self.memory_size]