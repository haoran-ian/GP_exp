import numpy as np

class AdaptiveHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30
        self.final_population_size = 15
        self.particles = np.random.rand(self.initial_population_size, dim)
        self.velocities = np.random.rand(self.initial_population_size, dim) * 0.1
        self.personal_best = self.particles.copy()
        self.global_best = None
        self.personal_best_scores = np.full(self.initial_population_size, np.inf)
        self.global_best_score = np.inf
        self.c1 = 2.0  # cognitive constant
        self.c2 = 2.0  # social constant
        self.w_initial = 0.9  # initial inertia weight
        self.w_final = 0.4    # final inertia weight
        self.mutation_factor_initial = 0.9
        self.mutation_factor_final = 0.5
        self.elitism_rate = 0.1  # percentage of top solutions retained

    def __call__(self, func):
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub
        eval_count = 0
        population_size = self.initial_population_size

        while eval_count < self.budget:
            # Adaptive inertia weight and mutation factor
            w = self.w_final + (self.w_initial - self.w_final) * (self.budget - eval_count) / self.budget
            mutation_factor = self.mutation_factor_final + (self.mutation_factor_initial - self.mutation_factor_final) * (self.budget - eval_count) / self.budget
            
            # Adaptive population size
            population_size = self.final_population_size + int((self.initial_population_size - self.final_population_size) * (self.budget - eval_count) / self.budget)
            self.particles = self.particles[:population_size]
            self.velocities = self.velocities[:population_size]
            self.personal_best = self.personal_best[:population_size]
            self.personal_best_scores = self.personal_best_scores[:population_size]

            # Evaluate current particles
            for i in range(population_size):
                position = self.particles[i]
                score = func(position)
                eval_count += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = position

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = position

            if eval_count >= self.budget:
                break

            # PSO: Update velocities and positions
            for i in range(population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] = (w * self.velocities[i] + 
                                      self.c1 * r1 * (self.personal_best[i] - self.particles[i]) + 
                                      self.c2 * r2 * (self.global_best - self.particles[i]))
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], bounds_lb, bounds_ub)

            # DE: Differential mutation with dynamic parameter
            for i in range(population_size):
                indices = np.random.choice(population_size, 3, replace=False)
                x0, x1, x2 = self.particles[indices]
                
                mutant_vector = x0 + mutation_factor * (x1 - x2)
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
            top_indices = np.argsort(self.personal_best_scores)[:int(self.elitism_rate * population_size)]
            elites = self.personal_best[top_indices]
            self.particles[top_indices] = elites

        return self.global_best