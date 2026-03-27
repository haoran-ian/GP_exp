import numpy as np

class EnhancedAPDES:  # Enhanced Advanced Phase-Dynamic Evolutionary Strategy
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 50
        mutation_factor = 0.8
        initial_crossover_rate = 0.7
        
        # Initialize population with increased diversity
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        while evaluations < self.budget:
            # Dynamic mutation and crossover rate based on phase
            phase = evaluations / self.budget
            if phase < 0.5:
                mutation_factor = 0.5 + phase
                crossover_rate = initial_crossover_rate - 0.4 * phase
            else:
                mutation_factor = max(0.5, 1.0 - phase)
                crossover_rate = 0.5 + 0.4 * (1.0 - phase)
            
            # Dynamic population resizing
            if evaluations % (self.budget // 10) == 0:
                population_size = max(20, int(population_size * (0.9 if phase < 0.5 else 1.1)))
                population = np.resize(population, (population_size, self.dim))
                fitness = np.resize(fitness, population_size)
            
            new_population = []
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                # Adjust mutation factor based on diversity
                diversity = np.std(population, axis=0).sum()
                adaptive_mutation_factor = mutation_factor * (1 + 0.01 * diversity)
                
                mutant_vector = a + adaptive_mutation_factor * (b - c)
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
            
            # Diversity-based selection and reinitialization
            diversity = np.std(new_population, axis=0).sum()
            if diversity < 1e-5:  # Trigger alternative mutation strategy
                for idx in range(population_size):
                    new_population[idx] = np.random.uniform(lb, ub, self.dim)
            
            # Elitism with diversity enhancement
            best_idx = np.argmin(fitness)
            if np.random.rand() < 0.1:
                random_idx = np.random.randint(population_size)
                new_population[random_idx] = population[best_idx]
            
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]