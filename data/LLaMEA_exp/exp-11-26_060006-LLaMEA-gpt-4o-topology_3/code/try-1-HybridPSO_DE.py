import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20  # A reasonable size for the population
        self.particles = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        self.velocities = np.random.uniform(
            -1.0, 1.0, (self.population_size, self.dim)
        )
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        w = 0.9  # Adaptive inertia weight
        c1 = 1.5  # Cognitive component
        c2 = 1.5  # Social component
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Evaluate current particle
                score = func(self.particles[i])
                self.evaluations += 1

                # Update personal best
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]

                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]

            # Update velocities and positions using PSO
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            cognitive_velocity = c1 * r1 * (self.personal_best_positions - self.particles)
            social_velocity = c2 * r2 * (self.global_best_position - self.particles)
            self.velocities = w * self.velocities + cognitive_velocity + social_velocity
            self.particles += self.velocities

            # Ensure particles are within bounds
            self.particles = np.clip(self.particles, self.lower_bound, self.upper_bound)

            # Apply differential evolution-like mutation and crossover
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                trial = np.copy(self.particles[i])
                
                # Binomial crossover
                for j in range(self.dim):
                    if np.random.rand() < CR:
                        trial[j] = mutant[j]
                
                trial_score = func(trial)
                self.evaluations += 1

                # Selection
                if trial_score < self.personal_best_scores[i]:
                    self.particles[i] = trial
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial

                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial

        return self.global_best_position