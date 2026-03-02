import numpy as np

class HSDAS:  # Harmony Search-based Dynamic Adaptive Strategy
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 50
        harmony_memory_size = 10
        harmony_memory = np.random.uniform(lb, ub, (harmony_memory_size, self.dim))
        harmony_fitness = np.array([func(hm) for hm in harmony_memory])
        evaluations = harmony_memory_size
        
        while evaluations < self.budget:
            # Dynamic parameter adjustment
            phase = evaluations / self.budget
            harmony_consideration_rate = 0.9 * (1 - phase) + 0.1
            pitch_adjustment_rate = 0.1 * phase
            
            new_population = []
            for _ in range(population_size):
                if np.random.rand() < harmony_consideration_rate:
                    candidate = harmony_memory[np.random.randint(harmony_memory_size)]
                else:
                    candidate = np.random.uniform(lb, ub, self.dim)
                
                # Pitch adjustment
                if np.random.rand() < pitch_adjustment_rate:
                    candidate += np.random.normal(0, 0.1, self.dim)
                    candidate = np.clip(candidate, lb, ub)
                
                new_population.append(candidate)
            
            # Evaluate new population
            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += population_size
            
            # Update harmony memory with elitism
            combined_population = np.vstack((harmony_memory, new_population))
            combined_fitness = np.hstack((harmony_fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:harmony_memory_size]
            harmony_memory = combined_population[best_indices]
            harmony_fitness = combined_fitness[best_indices]
        
        best_idx = np.argmin(harmony_fitness)
        return harmony_memory[best_idx]