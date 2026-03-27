import numpy as np
from scipy.stats import norm

class EnhancedHybridPSO_DE_Chaotic:
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
        self.w_initial = 0.9  # initial inertia weight
        self.w_final = 0.4    # final inertia weight
        self.mutation_factor = 0.8  # DE mutation factor
        self.elitism_rate = 0.1  # percentage of top solutions retained
        self.learning_rate = 0.01  # initial learning rate

    def chaotic_map(self, n):
        # Logistic map for generating chaotic sequence
        x = 0.7
        for _ in range(n):
            x = 4.0 * x * (1 - x)
        return x

    def __call__(self, func):
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub
        eval_count = 0

        while eval_count < self.budget:
            # Adaptive inertia weight
            w = self.w_final + (self.w_initial - self.w_final) * (self.budget - eval_count) / self.budget
            
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
            chaotic_factor = self.chaotic_map(eval_count)
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] = (w * self.velocities[i] + 
                                      self.c1 * r1 * (self.personal_best[i] - self.particles[i]) + 
                                      self.c2 * r2 * (self.global_best - self.particles[i]))
                self.particles[i] += self.velocities[i] * chaotic_factor
                self.particles[i] = np.clip(self.particles[i], bounds_lb, bounds_ub)

            # DE: Differential mutation with diversity preservation
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.particles[indices]
                
                mutant_vector = x0 + self.mutation_factor * (x1 - x2)
                mutant_vector = np.clip(mutant_vector, bounds_lb, bounds_ub)

                dynamic_crossover_rate = 0.9 - 0.5 * (eval_count / self.budget)
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

            # Elitism: Retain a portion of the best solutions
            if eval_count >= self.budget:
                break

            top_indices = np.argsort(self.personal_best_scores)[:int(self.elitism_rate * self.population_size)]
            elites = self.personal_best[top_indices]
            self.particles[top_indices] = elites

            # Adaptive learning rate adjustment
            self.learning_rate *= 0.99  # Decay learning rate

        return self.global_best