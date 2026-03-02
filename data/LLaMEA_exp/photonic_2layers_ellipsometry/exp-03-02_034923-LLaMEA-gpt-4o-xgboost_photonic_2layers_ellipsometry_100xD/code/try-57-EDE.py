import numpy as np

class EDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 50
        initial_mutation_factor = 0.8
        initial_crossover_rate = 0.7
        
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        while evaluations < self.budget:
            phase = evaluations / self.budget
            mutation_factor = initial_mutation_factor * (1 - phase / 2)
            crossover_rate = initial_crossover_rate * (1 - phase / 2)
            
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
            
            diversity = np.std(new_population, axis=0).sum()
            if diversity < 1e-5:
                additional_population = np.random.uniform(lb, ub, (population_size // 2, self.dim))
                new_population.extend(additional_population)
                new_population = np.array(new_population[:population_size])
                fitness = np.array([func(ind) for ind in new_population])
                evaluations += population_size // 2
            
            best_idx = np.argmin(fitness)
            new_population[np.random.randint(population_size)] = population[best_idx]
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]