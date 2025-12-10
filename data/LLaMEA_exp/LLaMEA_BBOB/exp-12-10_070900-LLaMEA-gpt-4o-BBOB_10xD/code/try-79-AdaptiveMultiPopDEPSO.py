import numpy as np

class AdaptiveMultiPopDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.main_pop_size = max(4, self.dim * 2)
        self.sub_pop_size = max(2, self.dim)
        self.lb = -5.0
        self.ub = 5.0
        self.main_population = np.random.uniform(self.lb, self.ub, (self.main_pop_size, dim))
        self.sub_population = np.random.uniform(self.lb, self.ub, (self.sub_pop_size, dim))
        self.main_velocities = np.random.uniform(-1, 1, (self.main_pop_size, dim))
        self.sub_velocities = np.random.uniform(-1, 1, (self.sub_pop_size, dim))
        self.main_best_positions = np.copy(self.main_population)
        self.sub_best_positions = np.copy(self.sub_population)
        self.main_global_best_position = None
        self.sub_global_best_position = None
        self.main_best_values = np.full(self.main_pop_size, np.inf)
        self.sub_best_values = np.full(self.sub_pop_size, np.inf)
        self.global_best_value = np.inf
        self.f_evals = 0

    def evaluate(self, func, pos):
        if self.f_evals < self.budget:
            value = func(pos)
            self.f_evals += 1
            return value
        return np.inf

    def update_positions(self, func, population, best_positions, best_values):
        for i in range(population.shape[0]):
            fitness_value = self.evaluate(func, population[i])
            if fitness_value < best_values[i]:
                best_values[i] = fitness_value
                best_positions[i] = population[i]

            if fitness_value < self.global_best_value:
                self.global_best_value = fitness_value
                if population is self.main_population:
                    self.main_global_best_position = population[i]
                else:
                    self.sub_global_best_position = population[i]

    def differential_evolution_step(self, func, population, best_positions, best_values):
        F_base = 0.8
        for i in range(population.shape[0]):
            indices = np.random.choice(population.shape[0], 3, replace=False)
            x1, x2, x3 = population[indices]
            F = F_base + 0.2 * (np.std(population) / 5.0)
            mutant = x1 + F * (x2 - x3)
            mutant = np.clip(mutant, self.lb, self.ub)
            crossover_rate = 0.9 * (1 - self.f_evals / self.budget)
            trial = np.where(np.random.rand(self.dim) < crossover_rate, mutant, population[i])
            trial_fitness = self.evaluate(func, trial)
            if trial_fitness < best_values[i]:
                population[i] = trial
                best_values[i] = trial_fitness

    def particle_swarm_optimization_step(self, population, velocities, best_positions, global_best_position):
        for i in range(population.shape[0]):
            r1, r2 = np.random.rand(2)
            inertia_weight = 0.5 + (0.5 * (1 - self.f_evals / self.budget))
            cognitive = 2.0 * r1 * (best_positions[i] - population[i])
            social = 2.0 * r2 * (global_best_position - population[i])
            scaling_factor = 1.2
            distances = np.linalg.norm(population - population[i], axis=1)
            crowding_distance = np.min(distances[distances > 0])
            velocity_control_factor = 1 + 0.5 * (1 - np.tanh(crowding_distance))
            velocities[i] = scaling_factor * (inertia_weight * velocities[i] + cognitive + social) * velocity_control_factor
            population[i] = np.clip(population[i] + velocities[i], self.lb, self.ub)

        if self.f_evals / self.budget > 0.5 and np.std(best_values) < 1e-5:
            population[:] = np.random.uniform(self.lb, self.ub, population.shape)

    def __call__(self, func):
        while self.f_evals < self.budget:
            self.update_positions(func, self.main_population, self.main_best_positions, self.main_best_values)
            self.update_positions(func, self.sub_population, self.sub_best_positions, self.sub_best_values)
            self.differential_evolution_step(func, self.main_population, self.main_best_positions, self.main_best_values)
            self.differential_evolution_step(func, self.sub_population, self.sub_best_positions, self.sub_best_values)
            self.particle_swarm_optimization_step(self.main_population, self.main_velocities, self.main_best_positions, self.main_global_best_position)
            self.particle_swarm_optimization_step(self.sub_population, self.sub_velocities, self.sub_best_positions, self.sub_global_best_position)
        return self.main_global_best_position if self.global_best_value <= min(self.sub_best_values) else self.sub_global_best_position, self.global_best_value