import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 15 * dim
        self.temp_schedule = lambda t: max(0.01, 1.0 - t / self.budget)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        global_best = population[best_idx]
        global_best_fitness = fitness[best_idx]
        
        evaluations = self.population_size
        initial_mutation_factor = 0.8

        while evaluations < self.budget:
            new_population = []
            diversity = np.mean(np.std(population, axis=0))
            dynamic_crossover_prob = max(0.5, min(1.0, 1.5 * diversity))
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                # Adaptive mutation scaling factor based on convergence progress
                progress = evaluations / self.budget
                mutation_factor = initial_mutation_factor * (1.0 - progress)
                
                mutant = np.clip(a + mutation_factor * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < dynamic_crossover_prob
                trial = np.where(crossover, mutant, population[i])
                
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temp_schedule(evaluations)):
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])

                if trial_fitness < global_best_fitness:
                    global_best = trial
                    global_best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            population = np.array(new_population)

        return global_best