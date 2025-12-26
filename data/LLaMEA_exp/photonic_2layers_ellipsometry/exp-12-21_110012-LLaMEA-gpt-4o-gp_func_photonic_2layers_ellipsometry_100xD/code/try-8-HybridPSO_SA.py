import numpy as np

class HybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(max(5, dim), 30)  # Adjust number of particles based on dimensionality
        self.w = 0.5  # Inertia weight for PSO
        self.c1 = 1.5  # Cognitive coefficient for PSO
        self.c2 = 1.5  # Social coefficient for PSO
        self.temp_initial = 10  # Initial temperature for SA
        self.temp_final = 0.1  # Final temperature for SA
        self.cooling_rate = 0.95  # Cooling rate for SA
        self.positions = None
        self.velocities = None
        self.best_individual_positions = None
        self.best_individual_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.function_evaluations = 0

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        bounds = func.bounds
        lb = bounds.lb
        ub = bounds.ub

        # Initialize particles
        self.positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.best_individual_positions = self.positions.copy()
        self.best_individual_scores = np.array([func(position) for position in self.positions])
        self.function_evaluations += self.num_particles

        self.global_best_position = self.best_individual_positions[np.argmin(self.best_individual_scores)]
        self.global_best_score = np.min(self.best_individual_scores)

        # Optimization loop
        while self.function_evaluations < self.budget:
            self.w = 0.9 - (self.function_evaluations / self.budget) * (0.9 - 0.4)  # Line change: adapt inertia weight

            # Update velocities and positions
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.best_individual_positions[i] - self.positions[i]) +
                                      self.c2 * r2 * (self.global_best_position - self.positions[i]))
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], lb, ub)

                # Evaluate new position
                score = func(self.positions[i])
                self.function_evaluations += 1

                # Update personal best
                if score < self.best_individual_scores[i]:
                    self.best_individual_scores[i] = score
                    self.best_individual_positions[i] = self.positions[i].copy()

                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()

            # Simulated annealing for exploration
            temp = self.temp_initial * (self.cooling_rate ** (self.function_evaluations / self.budget))
            if temp > self.temp_final:
                for i in range(self.num_particles):
                    candidate_position = self.positions[i] + np.random.normal(0, temp, self.dim)
                    candidate_position = np.clip(candidate_position, lb, ub)
                    candidate_score = func(candidate_position)
                    self.function_evaluations += 1

                    if candidate_score < self.best_individual_scores[i] or \
                       np.random.rand() < np.exp(-(candidate_score - self.best_individual_scores[i]) / temp):
                        self.positions[i] = candidate_position
                        self.best_individual_scores[i] = candidate_score
                        self.best_individual_positions[i] = candidate_position

                        if candidate_score < self.global_best_score:
                            self.global_best_score = candidate_score
                            self.global_best_position = candidate_position

        return self.global_best_position, self.global_best_score