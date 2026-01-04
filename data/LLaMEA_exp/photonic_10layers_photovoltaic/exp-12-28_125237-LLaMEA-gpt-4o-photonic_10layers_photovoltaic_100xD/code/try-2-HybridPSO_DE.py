import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.best_particle_positions = np.copy(self.particles)
        self.best_particle_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.fitness_evaluations = 0

    def __call__(self, func):
        while self.fitness_evaluations < self.budget:
            self.update_particles(func)
            self.apply_differential_evolution(func)

        return self.global_best_position

    def update_particles(self, func):
        for i in range(self.population_size):
            if self.fitness_evaluations >= self.budget:
                break
            
            # Evaluate fitness
            score = func(self.particles[i])
            self.fitness_evaluations += 1
            
            # Update personal best
            if score < self.best_particle_scores[i]:
                self.best_particle_scores[i] = score
                self.best_particle_positions[i] = self.particles[i].copy()
            
            # Update global best
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.particles[i].copy()
        
        # Update velocities and positions
        w = 0.9 - ((self.fitness_evaluations / self.budget) * 0.4)  # Adaptive inertia weight
        for i in range(self.population_size):
            r1, r2 = np.random.rand(2)
            self.velocities[i] = (w * self.velocities[i] +
                                  1.5 * r1 * (self.best_particle_positions[i] - self.particles[i]) +
                                  1.5 * r2 * (self.global_best_position - self.particles[i]))
            self.particles[i] += self.velocities[i]
            # Ensure particles are within bounds
            self.particles[i] = np.clip(self.particles[i], func.bounds.lb, func.bounds.ub)

    def apply_differential_evolution(self, func):
        for i in range(self.population_size):
            if self.fitness_evaluations >= self.budget:
                break

            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = self.particles[a] + 0.8 * (self.particles[b] - self.particles[c])
            mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

            # Crossover
            cross_points = np.random.rand(self.dim) < 0.9
            trial = np.where(cross_points, mutant, self.particles[i])

            # Evaluate trial vector
            score = func(trial)
            self.fitness_evaluations += 1

            # Selection
            if score < self.best_particle_scores[i]:
                self.particles[i] = trial
                self.best_particle_scores[i] = score
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = trial