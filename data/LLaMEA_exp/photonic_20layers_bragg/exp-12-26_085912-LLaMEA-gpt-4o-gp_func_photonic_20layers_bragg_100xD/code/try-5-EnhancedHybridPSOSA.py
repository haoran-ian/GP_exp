import numpy as np

class EnhancedHybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.temperature = 1.0
        self.alpha = 0.97
        self.best_global_position = None
        self.best_global_value = float('inf')
        self.current_evals = 0
        self.neighbor_radius = 0.1

    def __call__(self, func):
        particles = np.random.uniform(
            low=func.bounds.lb,
            high=func.bounds.ub,
            size=(self.population_size, self.dim)
        )
        velocities = np.random.uniform(
            low=-abs(func.bounds.ub - func.bounds.lb),
            high=abs(func.bounds.ub - func.bounds.lb),
            size=(self.population_size, self.dim)
        )
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.population_size, float('inf'))
        
        while self.current_evals < self.budget:
            for i in range(self.population_size):
                if self.current_evals >= self.budget:
                    break
                fitness_value = func(particles[i])
                self.current_evals += 1

                if fitness_value < personal_best_values[i]:
                    personal_best_values[i] = fitness_value
                    personal_best_positions[i] = particles[i]

                if fitness_value < self.best_global_value:
                    self.best_global_value = fitness_value
                    self.best_global_position = particles[i]

            w = self.w_max - ((self.w_max - self.w_min) * (self.current_evals / self.budget)**0.5)  # Adaptive inertia weight

            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    w * velocities[i]
                    + self.c1 * r1 * (personal_best_positions[i] - particles[i])
                    + self.c2 * r2 * (self.best_global_position - particles[i])
                )
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], func.bounds.lb, func.bounds.ub)

            for i in range(self.population_size):
                if self.current_evals >= self.budget:
                    break
                perturbation = np.random.uniform(-self.neighbor_radius, self.neighbor_radius, self.dim)
                candidate_position = particles[i] + perturbation * np.random.rand() * (func.bounds.ub - func.bounds.lb)  # Improved neighborhood search
                candidate_position = np.clip(candidate_position, func.bounds.lb, func.bounds.ub)
                candidate_fitness = func(candidate_position)
                self.current_evals += 1

                if candidate_fitness < personal_best_values[i]:
                    personal_best_values[i] = candidate_fitness
                    personal_best_positions[i] = candidate_position
                elif np.exp((personal_best_values[i] - candidate_fitness) / self.temperature) > np.random.rand():
                    personal_best_values[i] = candidate_fitness
                    personal_best_positions[i] = candidate_position

            self.temperature *= self.alpha
            self.neighbor_radius *= 0.99  # shrink neighborhood as optimization progresses
        
        return self.best_global_position