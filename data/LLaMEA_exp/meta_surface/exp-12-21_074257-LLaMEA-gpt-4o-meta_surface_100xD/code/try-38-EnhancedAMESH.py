import numpy as np

class EnhancedAMESH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = 10
        self.memory = []
        self.memory_decay_factor = 0.85  # Enhanced decay factor for rapid adaptation
        self.adapt_factor = 0.98  # Refined adaptation parameter
        self.phase_switch = budget // 2  # Switch from exploration to exploitation halfway
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10
        mutation_rate = 0.3
        sigma_init = 0.2
        diversity_threshold = 0.15
        
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= population_size

        self.update_memory(population, fitness)
        
        while self.budget > 0:
            if self.budget > self.phase_switch:
                # Exploration phase
                mutation_rate = max(mutation_rate * self.adapt_factor, diversity_threshold / (np.mean(np.std(population, axis=0)) + 1e-6))
            else:
                # Exploitation phase
                mutation_rate = min(mutation_rate / self.adapt_factor, 0.5)

            offspring = []
            for rank, parent in enumerate(population):
                if np.random.rand() < 0.5 or not self.memory:
                    memory_sample = self.memory[np.random.choice(len(self.memory))] if self.memory else np.zeros(self.dim)
                    direction = memory_sample - parent
                else:
                    direction = np.random.uniform(-1, 1, self.dim)
                
                adaptive_mutation_scale = (1 - (rank / population_size)) * mutation_rate
                direction += np.random.normal(0, sigma_init * adaptive_mutation_scale, self.dim)
                child = np.clip(parent + direction, lb, ub)
                offspring.append(child)
            
            offspring_fitness = np.array([func(ind) for ind in offspring])
            self.budget -= population_size
            
            combined_population = np.vstack((population, offspring))
            combined_fitness = np.hstack((fitness, offspring_fitness))
            best_indices = np.argsort(combined_fitness)[:population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]
            
            self.update_memory(population, fitness)
        
        best_index = np.argmin(fitness)
        return population[best_index]
    
    def update_memory(self, population, fitness):
        best_individuals = population[np.argsort(fitness)[:self.memory_size]]
        self.memory = [(self.memory_decay_factor * old + (1 - self.memory_decay_factor) * new) for old, new in zip(self.memory, best_individuals)]
        if len(self.memory) < self.memory_size:
            self.memory.extend(best_individuals[:self.memory_size - len(self.memory)])
        self.memory = self.memory[:self.memory_size]