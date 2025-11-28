import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.dynamic_population_size = lambda evaluations: max(5, int(self.population_size * (1 - (evaluations / self.budget) ** 1.2)))  # Change: Modified decay rate for dynamic population size
        self.particles = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        self.velocities = np.random.uniform(
            -0.5, 0.5, (self.population_size, self.dim)
        )
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        w = 0.9
        c1 = 1.3
        c2 = 1.7
        decay_rate = 0.95

        while self.evaluations < self.budget:
            current_population_size = self.dynamic_population_size(self.evaluations)
            for i in range(current_population_size):
                score = func(self.particles[i])
                self.evaluations += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]

            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            cognitive_velocity = c1 * r1 * (self.personal_best_positions - self.particles)
            social_velocity = c2 * r2 * (self.global_best_position - self.particles)
            self.velocities = (0.5 + 0.5 * (self.evaluations / self.budget)) * (self.velocities + cognitive_velocity + social_velocity)
            self.particles += self.velocities
            self.particles = np.clip(self.particles, self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                F = 0.6 + 0.4 * np.random.rand()
                CR = 0.9 + 0.05 * np.random.rand() * (self.budget - self.evaluations) / self.budget
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                trial = np.copy(self.particles[i])
                
                for j in range(self.dim):
                    if np.random.rand() < CR:
                        trial[j] = mutant[j]
                
                trial_score = func(trial)
                self.evaluations += 1

                if trial_score < self.personal_best_scores[i]:
                    self.particles[i] = trial
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial

                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial

            w *= (decay_rate ** (1 + 0.04 * (self.evaluations / self.budget)))
            c1 *= (1 - 0.01 * (self.evaluations / self.budget))
        return self.global_best_position