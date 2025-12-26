import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.w_max = 0.9  # initial inertia weight
        self.w_min = 0.4  # final inertia weight
        self.c1 = 1.5  # cognitive weight
        self.c2 = 1.8  # slightly increased social weight
        self.temperature = 1.0  # initial temperature for SA
        self.alpha = 0.95  # slightly increased cooling rate
        self.best_global_position = None
        self.best_global_value = float('inf')
        self.current_evals = 0

    def __call__(self, func):
        # Initialize particles
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
            # Evaluate current fitness
            for i in range(self.population_size):
                if self.current_evals >= self.budget:
                    break
                fitness_value = func(particles[i])
                self.current_evals += 1

                # Update personal best
                if fitness_value < personal_best_values[i]:
                    personal_best_values[i] = fitness_value
                    personal_best_positions[i] = particles[i]

                # Update global best
                if fitness_value < self.best_global_value:
                    self.best_global_value = fitness_value
                    self.best_global_position = particles[i]

            # Dynamic inertia weight
            w = self.w_max - ((self.w_max - self.w_min) * (self.current_evals / self.budget))

            # Update velocities and positions using PSO
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    w * velocities[i]
                    + self.c1 * r1 * (personal_best_positions[i] - particles[i])
                    + self.c2 * r2 * (self.best_global_position - particles[i])
                )
                particles[i] += velocities[i]
                # Clipping to remain within bounds
                particles[i] = np.clip(particles[i], func.bounds.lb, func.bounds.ub)

            # Apply Simulated Annealing perturbation
            for i in range(self.population_size):
                if self.current_evals >= self.budget:
                    break
                candidate_position = particles[i] + np.random.normal(0, self.temperature, self.dim)
                candidate_position = np.clip(candidate_position, func.bounds.lb, func.bounds.ub)
                candidate_fitness = func(candidate_position)
                self.current_evals += 1
                
                if candidate_fitness < personal_best_values[i]:
                    personal_best_values[i] = candidate_fitness
                    personal_best_positions[i] = candidate_position
                elif np.exp((personal_best_values[i] - candidate_fitness) / self.temperature) > np.random.rand():
                    personal_best_values[i] = candidate_fitness
                    personal_best_positions[i] = candidate_position

            # Cool down the temperature
            self.temperature *= self.alpha
        
        return self.best_global_position