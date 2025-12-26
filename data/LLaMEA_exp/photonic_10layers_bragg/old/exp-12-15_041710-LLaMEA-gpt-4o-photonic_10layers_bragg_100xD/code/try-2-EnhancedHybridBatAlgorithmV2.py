import numpy as np

class EnhancedHybridBatAlgorithmV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.frequency_min = 0
        self.frequency_max = 2
        self.loudness = np.random.uniform(0.5, 1.0, self.population_size)
        self.pulse_rate = np.random.uniform(0.2, 0.8, self.population_size)
        self.alpha = 0.98
        self.gamma = 0.95
        self.memory = np.zeros((self.population_size, self.dim))
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        eval_count = self.population_size
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                frequency = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.randn()
                velocities[i] += (population[i] - best_solution) * frequency
                candidate = population[i] + velocities[i]
                candidate = np.clip(candidate, lb, ub)
                
                if np.random.rand() > self.pulse_rate[i]:
                    candidate = best_solution + 0.01 * np.random.randn(self.dim) * self.loudness[i]
                
                candidate_fitness = func(candidate)
                eval_count += 1
                
                if candidate_fitness < fitness[i] and np.random.rand() < self.loudness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    self.memory[i] = candidate
                    self.loudness[i] *= self.alpha
                    self.pulse_rate[i] *= (1 - np.exp(-self.gamma * eval_count / self.budget))
                    
                if candidate_fitness < best_fitness:
                    best_solution = candidate
                    best_fitness = candidate_fitness
                    
                if eval_count >= self.budget:
                    break
            
            # Memory-enhanced exploration
            if eval_count < self.budget:
                for i in range(self.population_size):
                    if np.random.rand() < 0.1:
                        candidate_memory = self.memory[i] + 0.01 * np.random.randn(self.dim)
                        candidate_memory = np.clip(candidate_memory, lb, ub)
                        candidate_memory_fitness = func(candidate_memory)
                        eval_count += 1
                        
                        if candidate_memory_fitness < fitness[i]:
                            population[i] = candidate_memory
                            fitness[i] = candidate_memory_fitness
                        
                        if candidate_memory_fitness < best_fitness:
                            best_solution = candidate_memory
                            best_fitness = candidate_memory_fitness
                            
                        if eval_count >= self.budget:
                            break
        
        return best_solution