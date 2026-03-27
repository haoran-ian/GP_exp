import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.personal_best = self.particles.copy()
        self.global_best = None
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_score = np.inf
        self.c1 = 2.0  # cognitive constant
        self.c2 = 2.0  # social constant
        self.w = 0.9   # initial inertia weight

    def __call__(self, func):
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub
        eval_count = 0

        while eval_count < self.budget:
            # Adaptively adjust inertia weight
            self.w = 0.9 - (eval_count / self.budget) * 0.4
            
            # Evaluate current particles
            for i in range(self.population_size):
                position = self.particles[i]
                score = func(position)
                eval_count += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = position

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = position

            # PSO: Update velocities and positions
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] = (self.w * self.velocities[i] + 
                                      self.c1 * r1 * (self.personal_best[i] - self.particles[i]) + 
                                      self.c2 * r2 * (self.global_best - self.particles[i]))
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], bounds_lb, bounds_ub)

            # DE: Differential mutation and crossover
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.particles[indices]
                
                mutant_vector = x0 + 0.8 * (x1 - x2)
                mutant_vector = np.clip(mutant_vector, bounds_lb, bounds_ub)

                crossover_mask = np.random.rand(self.dim) < 0.9
                trial_vector = np.where(crossover_mask, mutant_vector, self.particles[i])

                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best[i] = trial_vector
                    self.particles[i] = trial_vector

                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best = trial_vector
            
            if eval_count >= self.budget:
                break

        return self.global_best