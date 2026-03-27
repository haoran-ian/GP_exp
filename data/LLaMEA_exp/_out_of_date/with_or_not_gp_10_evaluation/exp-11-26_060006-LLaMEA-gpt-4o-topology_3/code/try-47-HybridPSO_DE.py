import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20  # A reasonable size for the initial population
        self.dynamic_population_size = lambda evaluations: max(5, int(self.population_size * (1 - evaluations / self.budget)))  # Dynamic population size
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
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive component
        c2 = 1.5  # Social component
        decay_rate = 0.99  # New decay rate for inertia weight

        while self.evaluations < self.budget:
            current_population_size = self.dynamic_population_size(self.evaluations)  # Update population size
            for i in range(current_population_size):
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

            # Apply adaptive differential evolution-like mutation and crossover
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                F = 0.5 + 0.3 * np.random.rand()  # Adaptive differential weight
                CR = 0.8 + 0.1 * np.random.rand()  # Adaptive crossover probability
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

            if self.evaluations / self.budget > 0.8 and np.random.rand() < 0.1:  # Random restart
                self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, self.dim))

            w *= (decay_rate ** (1 + 0.06 * (self.evaluations / self.budget)))  # More aggressive dynamic decay rate
        return self.global_best_position