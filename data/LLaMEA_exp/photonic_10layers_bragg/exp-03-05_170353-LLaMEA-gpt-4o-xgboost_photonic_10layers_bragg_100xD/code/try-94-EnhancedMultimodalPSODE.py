import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import levy_stable

class EnhancedMultimodalPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim  # Adjusted population size for better exploration
        self.F_base = 0.6  # Modified base F for DE
        self.CR_base = 0.85  # Modified base CR for DE
        self.dynamic_niches_ratio = 0.25  # Adjusted niching ratio
        self.epsilon = 1e-4  # Adjusted convergence precision
        self.chaos_factor = 0.12  # Adjusted chaos factor for more stability
        self.pso_weight = 0.7  # Adjusted PSO inertia weight
        self.pso_c1 = 1.8  # Adjusted cognitive component
        self.pso_c2 = 2.1  # Adjusted social component
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))  # Adjusted initial velocities

    def adaptive_parameters(self, generation):
        chaotic_sequence = np.sin(generation * self.chaos_factor)
        F = self.F_base + chaotic_sequence * np.random.uniform(0.3, 0.7)
        CR = self.CR_base * chaotic_sequence + np.random.uniform(0.05, 0.25)
        return np.clip(F, 0.2, 0.9), np.clip(CR, 0.05, 0.95)

    def differential_evolution(self, func, bounds, population, generation):
        trial_population = np.copy(population)
        F, CR = self.adaptive_parameters(generation)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
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

    def adaptive_local_search(self, func, individual, fitness):
        grad_step = 1e-3
        gradient = np.zeros(self.dim)
        for d in range(self.dim):
            perturbed = np.copy(individual)
            perturbed[d] += grad_step
            gradient[d] = (func(perturbed) - fitness) / grad_step
        return individual - 0.02 * gradient  # Adjusted gradient descent step

    def dynamic_niching(self, population, fitness):
        niches = []
        niche_count = int(self.dynamic_niches_ratio * self.population_size)
        while len(niches) < niche_count:
            idx = np.random.choice(len(population))
            if not any(np.allclose(population[idx], niche, atol=0.05) for niche in niches):
                niches.append(population[idx])
        return niches

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
                    if np.random.rand() < 0.35:  # Levy flight with adjusted probability
                        population[i] = self.levy_flight(population[i])
                    else:
                        population[i] = self.adaptive_local_search(func, population[i], fitness[i])
                    fitness[i] = func(population[i])
                    evaluations += 1

            niches = self.dynamic_niching(population, fitness)
            distance_matrix = cdist(niches, population)
            for i in range(len(population)):
                if np.any(distance_matrix[:, i] < 0.1):
                    population[i] = np.random.uniform(bounds.lb, bounds.ub, self.dim)
                    fitness[i] = func(population[i])
                    evaluations += 1

            gbest_idx = np.argmin(fitness)
            gbest = population[gbest_idx]
            population, self.velocities = self.particle_swarm_update(population, pbest, gbest, self.velocities)

            for i in range(self.population_size):
                if fitness[i] < pbest_fitness[i]:
                    pbest[i] = population[i]
                    pbest_fitness[i] = fitness[i]

            generation += 1

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]