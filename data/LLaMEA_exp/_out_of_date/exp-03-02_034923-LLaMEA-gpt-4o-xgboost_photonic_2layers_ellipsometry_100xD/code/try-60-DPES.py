import numpy as np

class DPES:  # Dynamic Phase-Shifted Evolutionary Strategy
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 50
        initial_mutation_factor = 0.8
        initial_crossover_rate = 0.7
        
        # Initialize population with increased diversity
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.budget:
            # Dynamic phase-shifted mutation and crossover adaptation
            phase = evaluations / self.budget
            if phase < 0.3:
                mutation_factor = initial_mutation_factor + phase * 0.2
                crossover_rate = initial_crossover_rate - phase * 0.2
            elif phase < 0.6:
                mutation_factor = initial_mutation_factor + 0.3 * 0.2
                crossover_rate = initial_crossover_rate - 0.3 * 0.2
            else:
                mutation_factor = initial_mutation_factor - (phase - 0.6) * 0.5
                crossover_rate = initial_crossover_rate + (phase - 0.6) * 0.5
            
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
            
            # Diversity-based selection and phase-specific adjustment
            diversity = np.std(new_population, axis=0).sum()
            if diversity < 1e-5:  # Trigger alternative mutation strategy
                for idx in range(population_size):
                    new_population[idx] = np.random.uniform(lb, ub, self.dim)
            
            # Elitism with diversity enhancement
            best_idx = np.argmin(fitness)
            new_population[np.random.randint(population_size)] = population[best_idx]
            
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]