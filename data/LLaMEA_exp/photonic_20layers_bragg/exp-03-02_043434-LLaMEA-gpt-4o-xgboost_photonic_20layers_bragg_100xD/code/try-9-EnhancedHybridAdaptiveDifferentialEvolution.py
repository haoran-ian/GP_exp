import numpy as np

class EnhancedHybridAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 * dim
        self.population = np.random.rand(self.population_size, dim)
        self.F = 0.5
        self.CR = 0.9

    def levy_flight(self, L):
        u = np.random.normal(0, 1, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / L)
        return step

    def chaos_perturbation(self, x):
        # Logistic map for chaos-driven perturbation
        r = 4.0  # Chaos parameter
        return 4.0 * x * (1.0 - x)

    def differential_evolution(self, func, lb, ub):
        bounds = np.array([lb, ub])
        best_solution = None
        best_fitness = np.inf
        evaluations = 0

        self.population = lb + (ub - lb) * self.population
        fitness = np.apply_along_axis(func, 1, self.population)
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]

                # Multi-phase stochastic scaling for F and CR
                if evaluations < self.budget / 3:
                    F_dynamic = 0.5 + 0.2 * np.random.rand()
                    CR_dynamic = 0.9 - 0.1 * np.random.rand()
                elif evaluations < 2 * self.budget / 3:
                    F_dynamic = 0.7 + 0.1 * np.random.rand()
                    CR_dynamic = 0.8 - 0.05 * np.random.rand()
                else:
                    F_dynamic = 0.9 + 0.05 * np.random.rand()
                    CR_dynamic = 0.85 - 0.025 * np.random.rand()

                mutant_vector = np.clip(a + F_dynamic * (b - c), lb, ub)
                crossover_mask = np.random.rand(self.dim) < CR_dynamic
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                
                if np.random.rand() < 0.5:
                    trial_vector += self.levy_flight(1.5) * (trial_vector - self.population[i])

                # Apply chaos-driven perturbation
                trial_vector = self.chaos_perturbation(trial_vector)
                trial_vector = np.clip(trial_vector, lb, ub)
                
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