import numpy as np

class HybridSwarmOptimizer:
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
        self.f = 0.8  # Differential weight
        self.cr = 0.9  # Crossover probability

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub

        for eval_count in range(self.budget):
            # Evaluate particles
            for i in range(self.population_size):
                score = func(self.particles[i])

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = self.particles[i].copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.particles[i].copy()

            # Update velocities and positions for PSO
            inertia_weight = 0.9 - 0.5 * (eval_count / self.budget)  # Dynamic inertia weight
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

            # Differential Evolution mutation and crossover
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                self.f = 0.5 + 0.3 * np.random.rand()  # Adaptive differential weight
                mutant_vector = self.personal_best[a] + self.f * (self.personal_best[b] - self.personal_best[c])
                mutant_vector = np.clip(mutant_vector, lower_bound, upper_bound)
                
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

            # Ensure budget limit
            if eval_count >= self.budget:
                break

        return self.global_best