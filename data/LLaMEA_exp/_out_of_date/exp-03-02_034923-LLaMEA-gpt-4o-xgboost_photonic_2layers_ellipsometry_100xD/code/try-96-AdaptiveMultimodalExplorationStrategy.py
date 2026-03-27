import numpy as np

class AdaptiveMultimodalExplorationStrategy:  # AMES
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 50
        initial_mutation_factor = 0.8
        initial_crossover_rate = 0.7
        
        # Initialize an equally diverse population
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        while evaluations < self.budget:
            # Adaptive mutation factor and crossover based on exploration-exploitation balance
            phase_ratio = evaluations / self.budget
            mutation_factor = initial_mutation_factor * (1 - 0.5 * np.sin(np.pi * phase_ratio))
            crossover_rate = initial_crossover_rate * (0.6 + 0.4 * np.cos(np.pi * phase_ratio))
            
            # Adjust population size dynamically
            dynamic_factor = 0.75 + 0.5 * np.sin(np.pi * phase_ratio)
            population_size = int(max(20, population_size * dynamic_factor))
            population = np.resize(population, (population_size, self.dim))
            fitness = np.resize(fitness, population_size)
            
            new_population = []
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant_vector = a + mutation_factor * (b - c)
                mutant_vector = np.clip(mutant_vector, lb, ub)
                
                crossover = np.random.rand(self.dim) < crossover_rate
                trial_vector = np.where(crossover, mutant_vector, population[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1
                
                if trial_fitness < fitness[i]:
                    new_population.append(trial_vector)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])
            
            # Introduce adaptive reinitialization to maintain diversity
            diversity = np.std(new_population, axis=0).sum()
            if diversity < 1e-4:  # Trigger diversity boost
                for idx in range(population_size):
                    new_population[idx] = np.random.uniform(lb, ub, self.dim)

            # Elitism with periodic replacement
            best_idx = np.argmin(fitness)
            if np.random.rand() < 0.1 and evaluations % 20 == 0:
                random_idx = np.random.randint(population_size)
                new_population[random_idx] = population[best_idx]
            
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]