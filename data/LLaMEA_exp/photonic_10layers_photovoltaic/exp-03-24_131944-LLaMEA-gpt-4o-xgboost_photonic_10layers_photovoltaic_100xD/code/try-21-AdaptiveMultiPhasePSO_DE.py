import numpy as np

class AdaptiveMultiPhasePSO_DE:
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
        self.c1 = 1.5  # cognitive constant
        self.c2 = 1.5  # social constant
        self.w_initial = 0.9  # initial inertia weight
        self.w_final = 0.4    # final inertia weight
        self.mutation_factor = 0.8  # DE mutation factor
        self.elite_ratio = 0.1  # ratio of elite particles

    def __call__(self, func):
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub
        eval_count = 0
        elite_size = int(self.population_size * self.elite_ratio)

        while eval_count < self.budget:
            # Phase-dependent adaptive parameters
            phase_progress = eval_count / self.budget
            w = self.w_final + (self.w_initial - self.w_final) * (1 - phase_progress)
            mutation_factor = self.mutation_factor * (1 + 0.5 * np.cos(np.pi * phase_progress))

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
                self.velocities[i] = (w * self.velocities[i] + 
                                      self.c1 * r1 * (self.personal_best[i] - self.particles[i]) + 
                                      self.c2 * r2 * (self.global_best - self.particles[i]))
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], bounds_lb, bounds_ub)

            # DE: Differential mutation and crossover
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.particles[indices]
                
                mutant_vector = x0 + mutation_factor * (x1 - x2)
                mutant_vector = np.clip(mutant_vector, bounds_lb, bounds_ub)

                dynamic_crossover_rate = 0.9 - 0.5 * phase_progress
                crossover_mask = np.random.rand(self.dim) < dynamic_crossover_rate
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

            # Elite selection to preserve top individuals
            elite_indices = np.argsort(self.personal_best_scores)[:elite_size]
            elites = self.personal_best[elite_indices]
            self.particles[:elite_size] = elites

            if eval_count >= self.budget:
                break

        return self.global_best