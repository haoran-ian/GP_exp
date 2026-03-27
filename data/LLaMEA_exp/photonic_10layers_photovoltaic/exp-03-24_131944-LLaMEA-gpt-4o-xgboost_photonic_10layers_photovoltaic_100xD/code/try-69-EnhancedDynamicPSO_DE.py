import numpy as np

class EnhancedDynamicPSO_DE:
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
        self.c1_initial = 2.5  # initial cognitive constant
        self.c2_initial = 1.5  # initial social constant
        self.c1_final = 1.5    # final cognitive constant
        self.c2_final = 2.5    # final social constant
        self.w_initial = 0.9   # initial inertia weight
        self.w_final = 0.4     # final inertia weight
        self.mutation_factor = 0.9
        self.elitism_rate = 0.1
        self.local_search_prob = 0.1

    def __call__(self, func):
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub
        eval_count = 0

        while eval_count < self.budget:
            # Adaptive weights and constants
            t = eval_count / self.budget
            w = self.w_final + (self.w_initial - self.w_final) * (1 - t)
            c1 = self.c1_final + (self.c1_initial - self.c1_final) * (1 - t)
            c2 = self.c2_final + (self.c2_initial - self.c2_final) * (1 - t)
            
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

            # DE: Differential mutation with diversity preservation
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.particles[indices]
                
                mutant_vector = x0 + self.mutation_factor * (x1 - x2)
                mutant_vector = np.clip(mutant_vector, bounds_lb, bounds_ub)

                dynamic_crossover_rate = 0.9 - 0.5 * t
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
            top_indices = np.argsort(self.personal_best_scores)[:int(self.elitism_rate * self.population_size)]
            elites = self.personal_best[top_indices]
            self.particles[top_indices] = elites

            # Local search phase
            if np.random.rand() < self.local_search_prob:
                for i in top_indices:
                    local_position = self.particles[i] + np.random.uniform(-0.1, 0.1, self.dim)
                    local_position = np.clip(local_position, bounds_lb, bounds_ub)
                    local_score = func(local_position)
                    eval_count += 1

                    if local_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = local_score
                        self.personal_best[i] = local_position
                        self.particles[i] = local_position

                    if local_score < self.global_best_score:
                        self.global_best_score = local_score
                        self.global_best = local_position

            if eval_count >= self.budget:
                break

        return self.global_best