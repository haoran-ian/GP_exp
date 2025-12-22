import numpy as np

class EnhancedHybridSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.personal_best = self.particles.copy()
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best = None
        self.global_best_score = float('inf')
        self.initial_inertia = 0.9
        self.final_inertia = 0.4
        self.cr = 0.9  # Crossover probability

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        eval_count = 0

        while eval_count < self.budget:
            # Evaluate particles
            for i in range(self.population_size):
                score = func(self.particles[i])
                eval_count += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = self.particles[i].copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.particles[i].copy()

            # Update inertia weight linearly
            inertia_weight = self.initial_inertia - (eval_count / self.budget) * (self.initial_inertia - self.final_inertia)

            # Update velocities and positions for PSO
            cognitive_component = np.random.rand(self.population_size, self.dim)
            social_component = np.random.rand(self.population_size, self.dim)
            self.velocities = (
                inertia_weight * self.velocities
                + cognitive_component * (self.personal_best - self.particles)
                + social_component * (self.global_best - self.particles)
            )
            self.particles += self.velocities

            # Apply boundaries
            self.particles = np.clip(self.particles, lower_bound, upper_bound)

            # Enhanced Differential Evolution mutation and crossover
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                # Use an enhanced mutation strategy
                f = 0.5 + 0.5 * (np.random.rand() - 0.5)  # Adaptive factor
                mutant_vector = self.personal_best[a] + f * (self.personal_best[b] - self.personal_best[c])
                mutant_vector = np.clip(mutant_vector, lower_bound, upper_bound)

                # Leader-based crossover
                trial_vector = np.where(
                    np.random.rand(self.dim) < self.cr,
                    mutant_vector, 
                    self.particles[i]
                )

                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best[i] = trial_vector.copy()
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best = trial_vector.copy()

        return self.global_best