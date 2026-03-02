import numpy as np

class AMPEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 100
        mutation_factor = 0.5
        crossover_rate = 0.9
        
        # Initialize population with enhanced diversity
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        while evaluations < self.budget:
            new_population = []
            phase = evaluations / self.budget
            
            # Adaptive mutation and crossover rates
            if phase < 0.3:
                mutation_factor = 0.9
                crossover_rate = 0.6
            elif phase < 0.6:
                mutation_factor = 0.6
                crossover_rate = 0.8
            else:
                mutation_factor = 0.4
                crossover_rate = 0.9
            
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant_vector = a + mutation_factor * (b - c)
                mutant_vector = np.clip(mutant_vector, lb, ub)
                
                crossover = np.random.rand(self.dim) < crossover_rate
                trial_vector = np.where(crossover, mutant_vector, population[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1
                
                # Crowding-based selection
                if trial_fitness < fitness[i]:
                    new_population.append(trial_vector)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])
            
            # Random immigrant injection to maintain diversity
            if evaluations / self.budget > 0.8:
                num_immigrants = int(0.1 * population_size)
                immigrants = np.random.uniform(lb, ub, (num_immigrants, self.dim))
                new_population[-num_immigrants:] = immigrants
            
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]