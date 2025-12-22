import numpy as np

class EnhancedAMESH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_memory_size = 5
        self.max_memory_size = 15
        self.memory = []
        self.memory_decay_factor = 0.9
        self.adapt_factor = 0.99
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10
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
            mutation_rate = max(0.2 * self.adapt_factor, diversity_threshold / (diversity + 1e-6))
            
            offspring = []
            for rank, parent in enumerate(population):
                if np.random.rand() < 0.5 or not self.memory:
                    if self.memory:
                        memory_sample = self.memory[np.random.choice(len(self.memory))]
                        direction = memory_sample - parent
                    else:
                        direction = np.random.uniform(-1, 1, self.dim)
                else:
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
            
            # Update memory
            self.update_memory(population, fitness)
        
        # Return the best solution found
        best_index = np.argmin(fitness)
        return population[best_index]
    
    def update_memory(self, population, fitness):
        memory_size = min(self.max_memory_size, self.initial_memory_size + int(np.std(fitness) * 10))
        best_individuals = population[np.argsort(fitness)[:memory_size]]
        
        # Apply memory decay factor to reinforce new best individuals
        self.memory = [(self.memory_decay_factor * old + (1 - self.memory_decay_factor) * new) for old, new in zip(self.memory, best_individuals)]
        
        # Fill or trim memory to match the dynamic memory size
        if len(self.memory) < memory_size:
            self.memory.extend(best_individuals[:memory_size - len(self.memory)])
        else:
            self.memory = self.memory[:memory_size]