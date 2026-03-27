import numpy as np

class ADGO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 50
        mutation_factor = 0.8
        crossover_rate = 0.7
        
        # Initialize population with increased diversity
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        while evaluations < self.budget:
            # Adjust mutation and crossover rates dynamically
            phase = evaluations / self.budget
            mutation_factor = 0.6 + 0.4 * np.sin(phase * np.pi)
            crossover_rate = 0.4 + 0.6 * (1 - np.cos(phase * np.pi))
            
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
            
            # Enhance diversity with a modified mutation strategy
            diversity = np.std(new_population, axis=0).sum()
            if diversity < 1e-5:  # Trigger alternative mutation strategy
                for idx in range(population_size):
                    perturbation = np.random.normal(0, 0.1, self.dim)
                    new_population[idx] = np.clip(new_population[idx] + perturbation, lb, ub)
            
            # Elitism with diversity enhancement
            best_idx = np.argmin(fitness)
            new_population[np.random.randint(population_size)] = population[best_idx]
            
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]