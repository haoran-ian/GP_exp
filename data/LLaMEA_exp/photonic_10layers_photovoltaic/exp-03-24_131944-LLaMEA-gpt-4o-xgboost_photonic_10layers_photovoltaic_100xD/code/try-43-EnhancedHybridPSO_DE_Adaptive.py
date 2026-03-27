import numpy as np

class EnhancedHybridPSO_DE_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30
        self.particles = np.random.rand(self.initial_population_size, dim)
        self.velocities = np.random.rand(self.initial_population_size, dim) * 0.1
        self.personal_best = self.particles.copy()
        self.global_best = None
        self.personal_best_scores = np.full(self.initial_population_size, np.inf)
        self.global_best_score = np.inf
        self.c1_initial = 2.0
        self.c2_initial = 2.0
        self.w_initial = 0.9
        self.w_final = 0.4
        self.mutation_factor_initial = 0.8
        self.elitism_rate = 0.1
        self.population_size = self.initial_population_size

    def __call__(self, func):
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub
        eval_count = 0

        while eval_count < self.budget:
            # Adaptive inertia weight and adaptive learning rates
            w = self.w_final + (self.w_initial - self.w_final) * (self.budget - eval_count) / self.budget
            c1 = self.c1_initial * np.exp(-0.5 * eval_count / self.budget)
            c2 = self.c2_initial * np.exp(-0.5 * eval_count / self.budget)
            mutation_factor = self.mutation_factor_initial * (1 - eval_count / self.budget)

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
                                      c1 * r1 * (self.personal_best[i] - self.particles[i]) + 
                                      c2 * r2 * (self.global_best - self.particles[i]))
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], bounds_lb, bounds_ub)

            # DE: Differential mutation with dynamic crossover
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
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

                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best = trial_vector

            # Elitism: Retain a portion of the best solutions
            if eval_count >= self.budget:
                break

            top_indices = np.argsort(self.personal_best_scores)[:int(self.elitism_rate * self.population_size)]
            elites = self.personal_best[top_indices]
            self.particles[top_indices] = elites

            # Dynamic population size adjustment
            self.population_size = max(10, int(self.initial_population_size * (1 - eval_count / self.budget)))

        return self.global_best