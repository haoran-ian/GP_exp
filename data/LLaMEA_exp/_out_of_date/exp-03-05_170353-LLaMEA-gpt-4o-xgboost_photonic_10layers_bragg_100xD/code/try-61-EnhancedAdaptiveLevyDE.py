import numpy as np
from scipy.spatial.distance import cdist

class EnhancedAdaptiveLevyDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_base = 0.5
        self.CR_base = 0.9
        self.epsilon = 1e-3
        self.pso_weight = 0.7
        self.pso_c1 = 2.0
        self.pso_c2 = 2.0
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.subpopulation_count = 3
        self.subpopulations = [self.population_size // self.subpopulation_count] * self.subpopulation_count

    def levy_flight(self, step_size=0.1):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v)**(1 / beta)
        return step_size * step

    def adaptive_parameters(self, generation):
        chaotic_sequence = np.sin(generation * 0.1)
        F = self.F_base + chaotic_sequence * np.random.uniform(0.3, 0.8)
        CR = self.CR_base * chaotic_sequence + np.random.uniform(0.1, 0.3)
        return np.clip(F, 0.3, 0.8), np.clip(CR, 0.1, 1.0)

    def differential_evolution(self, func, bounds, population, generation):
        trial_population = np.copy(population)
        F, CR = self.adaptive_parameters(generation)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c) + self.levy_flight(), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial_population[i] = np.where(cross_points, mutant, population[i])
        return trial_population

    def local_search(self, func, individual):
        grad_step = 1e-3
        gradient = np.zeros(self.dim)
        for d in range(self.dim):
            perturbed = np.copy(individual)
            perturbed[d] += grad_step
            gradient[d] = (func(perturbed) - func(individual)) / grad_step
        return individual - 0.01 * gradient

    def particle_swarm_update(self, population, pbest, gbest, velocities):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        velocities = (self.pso_weight * velocities +
                      self.pso_c1 * r1 * (pbest - population) +
                      self.pso_c2 * r2 * (gbest - population))
        new_population = population + velocities
        return new_population, velocities

    def __call__(self, func):
        bounds = func.bounds
        population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        pbest = np.copy(population)
        pbest_fitness = np.copy(fitness)
        evaluations = self.population_size
        generation = 0

        while evaluations < self.budget:
            trial_population = self.differential_evolution(func, bounds, population, generation)
            trial_fitness = np.apply_along_axis(func, 1, trial_population)
            evaluations += self.population_size

            for i in range(self.population_size):
                if trial_fitness[i] + self.epsilon < fitness[i]:
                    population[i] = trial_population[i]
                    fitness[i] = trial_fitness[i]
                else:
                    population[i] = self.local_search(func, population[i])
                    fitness[i] = func(population[i])
                    evaluations += 1

            # Adaptive multi-population management
            subpopulations = np.array_split(population, self.subpopulation_count)
            subfitness = np.array_split(fitness, self.subpopulation_count)
            for sp, sf in zip(subpopulations, subfitness):
                gbest_idx = np.argmin(sf)
                gbest = sp[gbest_idx]
                sp, self.velocities = self.particle_swarm_update(sp, sp, gbest, self.velocities)
                new_fitness = np.apply_along_axis(func, 1, sp)
                evaluations += len(sp)
                for i in range(len(sp)):
                    if new_fitness[i] < sf[i]:
                        sf[i] = new_fitness[i]

            # Merge subpopulations back
            population = np.concatenate(subpopulations)
            fitness = np.concatenate(subfitness)

            generation += 1

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]