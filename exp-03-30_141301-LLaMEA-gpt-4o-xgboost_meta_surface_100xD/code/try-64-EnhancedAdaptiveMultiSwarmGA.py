import numpy as np
import scipy.spatial.distance as dist
from sklearn.decomposition import PCA

class EnhancedAdaptiveMultiSwarmGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50
        self.subswarms = 4
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.memory_size = 15
        self.memory = []
        self.evaluations = 0
        self.velocity_scale = 0.1

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        swarms = [self._initialize_swarm(bounds) for _ in range(self.subswarms)]
        velocities = [np.random.uniform(-self.velocity_scale, self.velocity_scale, (self.swarm_size, self.dim)) for _ in range(self.subswarms)]
        best_solution = None
        best_fitness = float('inf')

        while self.evaluations < self.budget:
            for idx, (swarm, velocity) in enumerate(zip(swarms, velocities)):
                fitness = np.apply_along_axis(func, 1, swarm)
                self.evaluations += len(fitness)
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < best_fitness:
                    best_fitness = fitness[best_idx]
                    best_solution = swarm[best_idx]
                    self._update_memory(best_solution)

                personal_best = swarm[np.argmin(fitness)]
                global_best = best_solution

                velocity = self._update_velocity(velocity, swarm, personal_best, global_best)
                swarm = self._update_positions(swarm, velocity, bounds)

                reduced_swarm = self._dimensionality_reduction(swarm)
                selected = self._selection(reduced_swarm, fitness)
                offspring = self._crossover(selected, bounds)
                swarm = self._mutation(offspring, bounds, idx)

                swarms[idx] = swarm
                velocities[idx] = velocity

        return best_solution

    def _initialize_swarm(self, bounds):
        return np.random.uniform(bounds[:, 0], bounds[:, 1], (self.swarm_size, self.dim))

    def _selection(self, swarm, fitness):
        selected_idx = np.random.choice(np.argsort(fitness)[:self.swarm_size // 2], size=self.swarm_size // 2, replace=True)
        return swarm[selected_idx]

    def _crossover(self, selected, bounds):
        offspring = []
        for i in range(self.swarm_size // 2):
            if np.random.rand() < self.crossover_rate:
                parent1, parent2 = selected[np.random.choice(len(selected), 2, replace=False)]
                cross_point = np.random.randint(1, self.dim - 1)
                child = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
                offspring.append(child)
            else:
                offspring.append(selected[i])
        return np.clip(offspring, bounds[:, 0], bounds[:, 1])

    def _mutation(self, offspring, bounds, subpop_idx):
        for i, individual in enumerate(offspring):
            if np.random.rand() < self.mutation_rate:
                mutation_vector = np.random.normal(0, 0.05, self.dim)
                offspring[i] = individual + mutation_vector
                offspring[i] = np.clip(offspring[i], bounds[:, 0], bounds[:, 1])
        return offspring

    def _update_memory(self, solution):
        if len(self.memory) < self.memory_size:
            self.memory.append(solution)
        else:
            improvement = np.min([np.linalg.norm(solution - mem) for mem in self.memory])
            if improvement > 0.1:
                self.memory[np.argmin([np.linalg.norm(solution - mem) for mem in self.memory])] = solution

    def _update_velocity(self, velocity, positions, personal_best, global_best):
        inertia = 0.5
        cognitive = 2 * np.random.rand() * (personal_best - positions)
        social = 2 * np.random.rand() * (global_best - positions)
        return inertia * velocity + cognitive + social

    def _update_positions(self, positions, velocity, bounds):
        new_positions = positions + velocity
        return np.clip(new_positions, bounds[:, 0], bounds[:, 1])

    def _dimensionality_reduction(self, population):
        if self.dim > 10:
            pca = PCA(n_components=10)  # Reduce to 10 dimensions or fewer
            reduced_population = pca.fit_transform(population)
            restored_population = pca.inverse_transform(reduced_population)
            return restored_population
        return population