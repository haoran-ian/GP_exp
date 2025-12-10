import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4, self.dim * 2)
        self.lb = -5.0
        self.ub = 5.0
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.best_positions = np.copy(self.population)
        self.global_best_position = None
        self.best_values = np.full(self.population_size, np.inf)
        self.global_best_value = np.inf
        self.f_evals = 0

    def evaluate(self, func, pos):
        if self.f_evals < self.budget:
            value = func(pos)
            self.f_evals += 1
            return value
        return np.inf

    def update_positions(self, func):
        for i in range(self.population_size):
            fitness_value = self.evaluate(func, self.population[i])
            if fitness_value < self.best_values[i]:
                self.best_values[i] = fitness_value
                self.best_positions[i] = self.population[i]

            if fitness_value < self.global_best_value:
                self.global_best_value = fitness_value
                self.global_best_position = self.population[i]

    def differential_evolution_step(self, func):
        F_base = 0.8
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            x1, x2, x3 = self.population[indices]
            F = F_base + 0.2 * (np.std(self.population) / 5.0)
            mutant = x1 + F * (x2 - x3)
            mutant = np.clip(mutant, self.lb, self.ub)
            crossover_rate = 0.9 * (1 - self.f_evals / self.budget)
            trial = np.where(np.random.rand(self.dim) < crossover_rate, mutant, self.population[i])
            trial_fitness = self.evaluate(func, trial)
            if trial_fitness < self.best_values[i]:
                self.population[i] = trial
                self.best_values[i] = trial_fitness

    def particle_swarm_optimization_step(self):
        for i in range(self.population_size):
            r1, r2 = np.random.rand(2)
            inertia_weight = 0.5 + (0.5 * (1 - self.f_evals / self.budget))
            cognitive = 2.0 * r1 * (self.best_positions[i] - self.population[i])
            social = 2.0 * r2 * (self.global_best_position - self.population[i])
            scaling_factor = 1.2

            distances = np.linalg.norm(self.population - self.population[i], axis=1)
            crowding_distance = np.min(distances[distances > 0])
            
            velocity_control_factor = 1 + 0.5 * (1 - np.tanh(crowding_distance))
            self.velocities[i] = scaling_factor * (inertia_weight * self.velocities[i] + cognitive + social) * velocity_control_factor
            self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lb, self.ub)

        if self.f_evals / self.budget > 0.5 and np.std(self.best_values) < 1e-5:
            self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

    def local_search_phase(self, func):
        if self.f_evals / self.budget > 0.75:  # Trigger local search towards the end
            for i in range(self.population_size):
                perturbation = np.random.normal(0, 0.1, self.dim)
                candidate = np.clip(self.global_best_position + perturbation, self.lb, self.ub)
                candidate_value = self.evaluate(func, candidate)
                if candidate_value < self.global_best_value:
                    self.global_best_value = candidate_value
                    self.global_best_position = candidate

    def __call__(self, func):
        while self.f_evals < self.budget:
            self.update_positions(func)
            self.differential_evolution_step(func)
            self.particle_swarm_optimization_step()
            self.local_search_phase(func)
        return self.global_best_position, self.global_best_value