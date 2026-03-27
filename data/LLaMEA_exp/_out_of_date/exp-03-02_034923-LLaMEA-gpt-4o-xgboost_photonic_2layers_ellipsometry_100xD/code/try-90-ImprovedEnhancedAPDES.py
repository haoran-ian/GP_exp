import numpy as np

class ImprovedEnhancedAPDES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 50
        mutation_factor = 0.8
        initial_crossover_rate = 0.7
        
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        while evaluations < self.budget:
            phase = evaluations / self.budget
            if phase < 0.3:
                mutation_factor = 0.9 - 0.4 * phase
                crossover_rate = initial_crossover_rate - 0.3 * phase
            elif phase < 0.7:
                mutation_factor = 0.5 + 0.5 * (phase - 0.3)
                crossover_rate = 0.4 + 0.6 * (phase - 0.3)
            else:
                mutation_factor = max(0.4, 0.8 - 0.5 * (phase - 0.7))
                crossover_rate = 0.7 + 0.2 * (1.0 - phase)
            
            if evaluations % (self.budget // 10) == 0:
                population_size = max(20, int(population_size * (0.9 if phase < 0.5 else 1.1)))
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
                
                adaptive_pressure = 1.0 - (evaluations / self.budget) ** 2
                if trial_fitness < fitness[i] * adaptive_pressure:
                    new_population.append(trial_vector)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])
            
            diversity = np.std(new_population, axis=0).sum()
            if diversity < 1e-5:
                for idx in range(population_size):
                    new_population[idx] = np.random.uniform(lb, ub, self.dim)
            
            best_idx = np.argmin(fitness)
            if np.random.rand() < 0.1:
                random_idx = np.random.randint(population_size)
                new_population[random_idx] = population[best_idx]
            
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]