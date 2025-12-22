import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.population_size = self.initial_population_size
        self.w = 0.5  # inertia weight
        self.c1 = 1.0  # cognitive parameter
        self.c2 = 2.0  # social parameter
        self.F = 0.5  # DE scaling factor
        self.CR = 0.9  # DE crossover probability

    def adapt_parameters(self, iteration, max_iterations):
        # Dynamically adjust parameters based on iteration progress
        progress = iteration / max_iterations
        self.w = 0.4 + (0.1 * np.cos(np.pi * progress))
        self.F = 0.4 + (0.1 * np.sin(np.pi * progress))
        self.CR = 0.9 - (0.5 * np.sin(progress * np.pi))  # removing progress multiplier for more adaptive CR

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)

        # Initialize the positions and velocities of particles
        particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) / 5
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size
        max_iterations = self.budget // self.population_size
        iteration = 0

        while evaluations < self.budget:
            # Update each particle's position using PSO and DE strategies
            for i in range(self.population_size):
                # PSO update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (global_best_position - particles[i]))

                # DE Mutation and Crossover
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = particles[indices]
                mutant_vector = x1 + self.F * (x2 - x3)
                crossover = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover, mutant_vector, particles[i])

                # Boundary handling
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
                trial_vector = np.clip(trial_vector, lb, ub)

                # Evaluate the trial vector
                trial_score = func(trial_vector)
                evaluations += 1

                # Update personal best
                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score

                    # Update global best
                    if trial_score < global_best_score:
                        global_best_position = trial_vector
                        global_best_score = trial_score

                if evaluations >= self.budget:
                    break

            # Dynamic population resizing and adaptive parameter tuning
            if iteration % (max_iterations // 3) == 0:
                if evaluations < (2 * self.budget) / 3:
                    self.population_size = max(10, self.initial_population_size // 2)
                else:
                    self.population_size = max(5, self.initial_population_size // 4)

            self.adapt_parameters(iteration, max_iterations)
            iteration += 1

        return global_best_position, global_best_score