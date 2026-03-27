import numpy as np

class HybridAdaptiveDifferentialEvolutionWithCompetition:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 * dim  # Adaptive population size
        self.population = np.random.rand(self.population_size, dim)
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def levy_flight(self, L):
        u = np.random.normal(0, 1, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / L)
        return step

    def competitive_selection(self, fitness):
        indices = np.argsort(fitness)
        return indices[:self.population_size // 2]  # Select the top half

    def differential_evolution(self, func, lb, ub):
        bounds = np.array([lb, ub])
        best_solution = None
        best_fitness = np.inf
        evaluations = 0

        self.population = lb + (ub - lb) * self.population
        fitness = np.apply_along_axis(func, 1, self.population)
        
        while evaluations < self.budget:
            selected_indices = self.competitive_selection(fitness)
            for i in selected_indices:
                indices = [idx for idx in selected_indices if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]

                # Introduce adaptive mutation scale
                F_adaptive = self.F + 0.2 * (fitness[i] - best_fitness) / (np.std(fitness) + 1e-9)
                F_adaptive = np.clip(F_adaptive, 0.1, 0.9)
                CR_dynamic = 0.9 - 0.2 * np.random.rand()
                
                mutant_vector = np.clip(a + F_adaptive * (b - c), lb, ub)
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

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution, best_fitness = self.differential_evolution(func, lb, ub)
        return best_solution