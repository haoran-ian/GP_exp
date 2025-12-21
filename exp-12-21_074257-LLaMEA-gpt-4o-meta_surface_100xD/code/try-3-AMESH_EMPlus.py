import numpy as np

class AMESH_EMPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = 5
        self.memory = []
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10
        mutation_rate = 0.1
        sigma_init = 0.5
        
        # Initialize population
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= population_size

        # Initialize memory
        self.memory.extend(population[np.argsort(fitness)[:self.memory_size]])
        
        while self.budget > 0:
            # Two-phase strategy based on budget utilization
            if self.budget > self.budget / 2:
                phase = "exploration"
            else:
                phase = "exploitation"
            
            offspring = []
            for _ in range(population_size):
                parent = population[np.random.choice(population_size)]
                memory_sample = self.memory[np.random.choice(len(self.memory))]
                direction = memory_sample - parent
                if np.random.rand() < mutation_rate:
                    if phase == "exploration":
                        mutation_scale = np.random.uniform(0.8, 2.0)
                    else:
                        mutation_scale = np.random.uniform(0.3, 1.2)
                    direction += np.random.normal(0, sigma_init * mutation_scale, self.dim)
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
            for new_best in population[:2]:
                worst_index = np.argmax([func(mem) for mem in self.memory])
                if func(new_best) < func(self.memory[worst_index]):
                    self.memory[worst_index] = new_best
        
        # Return the best solution found
        best_index = np.argmin(fitness)
        return population[best_index]