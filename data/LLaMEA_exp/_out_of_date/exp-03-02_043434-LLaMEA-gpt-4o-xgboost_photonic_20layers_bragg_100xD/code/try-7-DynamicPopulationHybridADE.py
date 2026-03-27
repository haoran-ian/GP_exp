import numpy as np

class DynamicPopulationHybridADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20 * dim
        self.population = np.random.rand(self.initial_pop_size, dim)
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def levy_flight(self, L):
        u = np.random.normal(0, 1, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / L)
        return step

    def differential_evolution(self, func, lb, ub):
        bounds = np.array([lb, ub])
        best_solution = None
        best_fitness = np.inf
        evaluations = 0

        self.population = lb + (ub - lb) * self.population
        fitness = np.apply_along_axis(func, 1, self.population)

        while evaluations < self.budget:
            if evaluations % (self.budget // 10) == 0:  # Dynamic resizing every 10% of the budget
                self.population = self.dynamic_resize(self.population, fitness)

            for i in range(len(self.population)):
                indices = [idx for idx in range(len(self.population)) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]

                # Dynamic adaptation for F and CR
                F_dynamic = 0.5 + 0.3 * np.random.rand()
                CR_dynamic = 0.9 - 0.2 * np.random.rand()
                
                mutant_vector = np.clip(a + F_dynamic * (b - c), lb, ub)
                crossover_mask = np.random.rand(self.dim) < CR_dynamic
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                
                if np.random.rand() < 0.5:  # Incorporate Lévy flights
                    trial_vector += self.levy_flight(1.5) * (trial_vector - self.population[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1
                
                if trial_fitness < fitness[i]:
                    self.population[i] = trial_vector
                    fitness[i] = trial_fitness

                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial_vector

                if evaluations >= self.budget:
                    break

        return best_solution, best_fitness

    def dynamic_resize(self, population, fitness):
        # Sort population based on fitness and retain half the population if budget allows
        sorted_indices = np.argsort(fitness)
        new_size = max(10, len(population) // 2)  # Never let the population fall below 10
        return population[sorted_indices][:new_size]

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution, best_fitness = self.differential_evolution(func, lb, ub)
        return best_solution