import numpy as np

class ADPE:  # Adaptive Diversity-Preserving Evolution
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
            phase = evaluations / self.budget
            if phase < 0.5:
                mutation_factor = min(1.0, 0.6 + phase * 0.8)
            else:
                crossover_rate = max(0.2, 1.0 - phase)
            
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
            
            # Diversity-based selection
            diversity = np.std(new_population, axis=0).sum()
            if diversity < 1e-5:  # Trigger alternative mutation strategy
                for idx in range(population_size):
                    new_population[idx] = np.random.uniform(lb, ub, self.dim)
            
            # Elitism with diversity enhancement
            best_idx = np.argmin(fitness)
            random_worst_idx = np.argmax(fitness)
            new_population[random_worst_idx] = population[best_idx]
            
            # Ensure a diverse set of solutions
            if evaluations < self.budget * 0.7:
                for idx in range(population_size):
                    if np.random.rand() < 0.1:
                        new_population[idx] = np.random.uniform(lb, ub, self.dim)

            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]