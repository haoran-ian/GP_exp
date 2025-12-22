import numpy as np

class AMESH_ANSS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = 5
        self.memory = []
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10
        mutation_rate = 0.15
        sigma_init = 0.5
        neighborhood_size = int(self.dim / 3)  # Adaptive neighborhood size
        
        # Initialize population
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= population_size

        # Initialize memory
        if len(self.memory) < self.memory_size:
            self.memory.extend(population[np.argsort(fitness)[:self.memory_size]])
        
        while self.budget > 0:
            # Evolutionary Strategy with adaptive memory
            offspring = []
            for _ in range(population_size):
                parent = population[np.random.choice(population_size)]
                memory_sample = self.memory[np.random.choice(len(self.memory))]
                direction = memory_sample - parent
                # Adaptive Neighborhood Search Strategy
                neighborhood_indices = np.random.choice(self.dim, neighborhood_size, replace=False)
                if np.random.rand() < mutation_rate:
                    for idx in neighborhood_indices:
                        direction[idx] += np.random.normal(0, sigma_init)
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
            if len(self.memory) < self.memory_size:
                self.memory.extend(population[:self.memory_size - len(self.memory)])
            else:
                for new_best in population[:3]:
                    worst_index = np.argmax([func(mem) for mem in self.memory])
                    if func(new_best) < func(self.memory[worst_index]):
                        self.memory[worst_index] = new_best
        
        # Return the best solution found
        best_index = np.argmin(fitness)
        return population[best_index]