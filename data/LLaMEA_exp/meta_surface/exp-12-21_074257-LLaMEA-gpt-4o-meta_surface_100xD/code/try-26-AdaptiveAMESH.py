import numpy as np

class AdaptiveAMESH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = 10
        self.memory = []
        self.memory_decay_factor = 0.8  # Increased decay factor for fresher memory
        self.adapt_factor = 0.98  # Enhanced adaptation for mutation scale
        self.dynamic_mutation = True  # Flag to toggle dynamic mutation strategies
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 12  # Increased size for more exploration
        mutation_rate = 0.25  # Enhanced starting mutation rate
        sigma_init = 0.4  # Larger initial sigma for more diverse exploration
        diversity_threshold = 0.05  # Lower threshold to encourage exploration
        
        # Initialize population
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= population_size
        
        # Initial memory update
        self.update_memory(population, fitness)
        
        while self.budget > 0:
            # Calculate diversity and adjust mutation rate dynamically
            diversity = np.mean(np.std(population, axis=0))
            if self.dynamic_mutation:
                mutation_rate = max(mutation_rate * self.adapt_factor, diversity_threshold / (diversity + 1e-6))
            
            offspring = []
            for rank, parent in enumerate(population):
                if np.random.rand() < 0.6 or not self.memory:  # Increased chance of using memory
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
            
            # Memory update
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