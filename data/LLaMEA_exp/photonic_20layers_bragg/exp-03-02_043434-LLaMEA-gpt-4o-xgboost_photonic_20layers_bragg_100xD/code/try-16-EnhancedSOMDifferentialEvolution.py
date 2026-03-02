import numpy as np
from minisom import MiniSom

class EnhancedSOMDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 * dim
        self.population = np.random.rand(self.population_size, dim)
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def levy_flight(self, L):
        u = np.random.normal(0, 1, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / L)
        return step

    def chaotic_local_search(self, position, lb, ub, chaos_level=0.1):
        chaotic_step = chaos_level * (np.random.rand(self.dim) - 0.5) * (ub - lb)
        new_position = position + chaotic_step
        return np.clip(new_position, lb, ub)

    def multi_scale_chaotic_search(self, position, lb, ub):
        for scale in [0.05, 0.1, 0.2]:
            position = self.chaotic_local_search(position, lb, ub, chaos_level=scale)
        return position

    def differential_evolution(self, func, lb, ub):
        bounds = np.array([lb, ub])
        best_solution = None
        best_fitness = np.inf
        evaluations = 0

        self.population = lb + (ub - lb) * self.population
        fitness = np.apply_along_axis(func, 1, self.population)

        som = MiniSom(3, 3, self.dim, sigma=0.3, learning_rate=0.5)
        som.random_weights_init(self.population)

        while evaluations < self.budget:
            som.train_random(self.population, 100)
            clusters = [som.winner(p) for p in self.population]

            for i in range(self.population_size):
                neighbors = [j for j in range(self.population_size) if clusters[j] == clusters[i]]
                if len(neighbors) < 3:
                    continue

                a, b, c = self.population[np.random.choice(neighbors, 3, replace=False)]

                F_dynamic = 0.5 + 0.3 * np.random.rand()
                CR_dynamic = 0.9 - 0.2 * np.random.rand()

                mutant_vector = np.clip(a + F_dynamic * (b - c), lb, ub)
                crossover_mask = np.random.rand(self.dim) < CR_dynamic
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])

                if np.random.rand() < 0.5:
                    trial_vector += self.levy_flight(1.5) * (trial_vector - self.population[i])

                trial_vector = self.multi_scale_chaotic_search(trial_vector, lb, ub)

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