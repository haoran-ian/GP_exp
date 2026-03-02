import numpy as np

class DynamicAdaptiveES:  # Dynamic Adaptive Evolutionary Strategy
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 50
        initial_mutation_factor = 0.7
        initial_crossover_rate = 0.9
        
        # Initialize population with increased diversity
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        while evaluations < self.budget:
            # Phase-aware adaption of mutation and crossover
            phase = evaluations / self.budget
            mutation_factor = initial_mutation_factor * (1 - 0.5 * np.sin(np.pi * phase))
            crossover_rate = initial_crossover_rate * (1 + 0.5 * np.cos(np.pi * phase))
            
            # Adaptive population resizing
            if evaluations % (self.budget // 5) == 0:
                population_size = max(20, int(population_size * (0.9 if phase < 0.5 else 1.1)))
                population = np.resize(population, (population_size, self.dim))
                fitness = np.resize(fitness, population_size)
            
            new_population = []
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                # Mutation with boundary check and corrective strategy
                mutant_vector = np.clip(a + mutation_factor * (b - c), lb, ub)
                
                # Crossover operation
                crossover = np.random.rand(self.dim) < crossover_rate
                trial_vector = np.where(crossover, mutant_vector, population[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1
                
                # Selection mechanism
                if trial_fitness < fitness[i]:
                    new_population.append(trial_vector)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])
            
            # Diversity preservation and elitism
            diversity = np.std(new_population, axis=0).sum()
            if diversity < 1e-4:
                for idx in range(population_size):
                    new_population[idx] = np.random.uniform(lb, ub, self.dim)
            
            best_idx = np.argmin(fitness)
            if np.random.rand() < 0.1:
                random_idx = np.random.randint(population_size)
                new_population[random_idx] = population[best_idx]
            
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]