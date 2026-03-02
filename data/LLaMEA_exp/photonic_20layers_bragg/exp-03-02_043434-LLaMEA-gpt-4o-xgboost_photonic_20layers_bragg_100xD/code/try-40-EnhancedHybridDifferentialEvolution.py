import numpy as np

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 * dim  # Initial population size
        self.population = np.random.rand(self.population_size, dim)
        self.initial_population_size = self.population_size
        self.F_base = 0.5  # Base differential weight
        self.CR_base = 0.9  # Base crossover probability
        self.inertia_weight = 0.9  # Inertia weight for adaptive control

    def levy_flight(self, L, step_size=0.01):  # Add step_size parameter
        u = np.random.normal(0, 1, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / L)
        return step_size * step  # Scale step by step_size

    def chaotic_local_search(self, position, lb, ub, chaos_level=0.1):
        chaotic_step = chaos_level * (np.random.rand(self.dim) - 0.5) * (ub - lb)
        new_position = position + chaotic_step
        return np.clip(new_position, lb, ub)

    def differential_evolution(self, func, lb, ub):
        bounds = np.array([lb, ub])
        best_solution = None
        best_fitness = np.inf
        evaluations = 0

        self.population = lb + (ub - lb) * self.population
        fitness = np.apply_along_axis(func, 1, self.population)

        while evaluations < self.budget:
            self.population_size = max(int(self.initial_population_size * (1 - evaluations / self.budget)), 4)
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]

                # Dynamic adaptation for F and CR with inertia weight
                F_dynamic = self.F_base + self.inertia_weight * np.random.rand()
                CR_dynamic = self.CR_base - self.inertia_weight * np.random.rand()

                mutant_vector = np.clip(a + F_dynamic * (b - c), lb, ub)
                crossover_mask = np.random.rand(self.dim) < CR_dynamic
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])

                if np.random.rand() < 0.5:  # Incorporate Lévy flights
                    trial_vector += self.levy_flight(1.5, step_size=0.02) * (trial_vector - self.population[i])

                # Apply chaotic local search to trial vector
                trial_vector = self.chaotic_local_search(trial_vector, lb, ub)

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