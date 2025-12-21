import numpy as np

class EnhancedAMESH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = 15
        self.memory = []
        self.memory_decay_factor = 0.85
        self.adapt_factor = 0.98
        self.crossover_rate = 0.7
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 12
        mutation_rate = 0.15
        sigma_init = 0.25
        diversity_threshold = 0.05
        
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= population_size

        self.update_memory(population, fitness)
        
        while self.budget > 0:
            diversity = np.mean(np.std(population, axis=0))
            mutation_rate = max(mutation_rate * self.adapt_factor, diversity_threshold / (diversity + 1e-6))
            
            offspring = []
            for rank, parent in enumerate(population):
                if np.random.rand() < self.crossover_rate and len(self.memory) >= 2:
                    parent2, parent3 = np.random.choice(self.memory, 2, replace=False)
                    direction = (parent2 + parent3) / 2 - parent
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
        else:
            self.memory = self.memory[:self.memory_size]