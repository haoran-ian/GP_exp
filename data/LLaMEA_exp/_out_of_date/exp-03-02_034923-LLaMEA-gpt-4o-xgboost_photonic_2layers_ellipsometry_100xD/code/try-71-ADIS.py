import numpy as np

class ADIS:
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
            if phase < 0.3:
                mutation_factor = min(1.2, 0.6 + phase)
                crossover_rate = max(0.7, 1.0 - phase)
            elif phase < 0.7:
                mutation_factor = max(0.5, 1.0 - phase)
                crossover_rate = max(0.5, phase)
            else:
                mutation_factor = max(0.3, 0.8 - phase)
                crossover_rate = min(0.9, 0.5 + phase)

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

            # Diversity control through standard deviation checking
            diversity = np.std(new_population, axis=0).sum()
            if diversity < 1e-5:
                for idx in range(population_size):
                    new_population[idx] = np.random.uniform(lb, ub, self.dim)

            # Enhanced elitism with diversity promotion
            best_idx = np.argmin(fitness)
            new_population[np.random.randint(population_size)] = population[best_idx]

            population = np.array(new_population)

        best_idx = np.argmin(fitness)
        return population[best_idx]