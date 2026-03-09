import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import levy_stable

class HybridMultiPopLevyStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.subpop_count = 4
        self.subpop_size = self.population_size // self.subpop_count
        self.F_base = 0.5
        self.CR_base = 0.9
        self.epsilon = 1e-3
        self.chaos_factor = 0.1
        self.pso_c1 = 2.0
        self.pso_c2 = 2.0
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))

    def adaptive_parameters(self, generation):
        chaotic_sequence = np.sin(generation * self.chaos_factor)
        F = self.F_base + chaotic_sequence * np.random.uniform(0.4, 0.8)
        CR = self.CR_base * chaotic_sequence + np.random.uniform(0.1, 0.3)
        return np.clip(F, 0.3, 0.8), np.clip(CR, 0.1, 1.0)

    def differential_evolution(self, func, bounds, population, generation):
        trial_population = np.copy(population)
        F, CR = self.adaptive_parameters(generation)
        for i in range(len(population)):
            idxs = [idx for idx in range(len(population)) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial_population[i] = np.where(cross_points, mutant, population[i])
        return trial_population

    def levy_flight(self, individual, alpha=1.5):
        levy_steps = levy_stable.rvs(alpha, 0, size=self.dim)
        return individual + 0.01 * levy_steps

    def local_search(self, func, individual, fitness):
        grad_step = 1e-3
        gradient = np.zeros(self.dim)
        for d in range(self.dim):
            perturbed = np.copy(individual)
            perturbed[d] += grad_step
            gradient[d] = (func(perturbed) - fitness) / grad_step
        return individual - 0.01 * gradient

    def cooperative_update(self, population, func, bounds):
        subpopulations = np.split(population, self.subpop_count)
        best_positions = []

        for subpop in subpopulations:
            fitness = np.apply_along_axis(func, 1, subpop)
            best_idx = np.argmin(fitness)
            best_positions.append(subpop[best_idx])

        new_pop = np.vstack(subpopulations)
        np.random.shuffle(new_pop)
        return new_pop

    def __call__(self, func):
        bounds = func.bounds
        population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size
        generation = 0

        while evaluations < self.budget:
            trial_population = self.differential_evolution(func, bounds, population, generation)
            trial_fitness = np.apply_along_axis(func, 1, trial_population)
            evaluations += len(trial_population)

            for i in range(len(population)):
                if trial_fitness[i] + self.epsilon < fitness[i]:
                    population[i] = trial_population[i]
                    fitness[i] = trial_fitness[i]
                else:
                    if np.random.rand() < 0.3:  
                        population[i] = self.levy_flight(population[i])
                    else:
                        population[i] = self.local_search(func, population[i], fitness[i])
                    fitness[i] = func(population[i])
                    evaluations += 1

            population = self.cooperative_update(population, func, bounds)
            generation += 1

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]