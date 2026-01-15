import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 15 * dim  # Typical for Differential Evolution
        self.temp_schedule = lambda t: max(0.01, 1.0 - t / self.budget)  # Cooling schedule for Simulated Annealing
        self.base_scaling_factor = 0.8
        self.adaptive_crossover_rate = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        global_best = population[best_idx]
        global_best_fitness = fitness[best_idx]
        
        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = []
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                scaling_factor = self.base_scaling_factor * (1 - evaluations/self.budget)  # Dynamic scaling factor
                mutant = np.clip(a + scaling_factor * (b - c), lb, ub)
                crossover_rate = self.adaptive_crossover_rate * (global_best_fitness / (fitness[i] + 1e-9))  # Adaptive crossover
                crossover = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover, mutant, population[i])
                
                # Simulated Annealing acceptance criterion
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