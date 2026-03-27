import numpy as np

class DynamicPopulationAdaptiveDE:
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
        self.c1 = 2.1
        self.c2 = 2.0
        self.w_initial = 0.9
        self.w_final = 0.4
        self.elitism_rate = 0.1

    def __call__(self, func):
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub
        eval_count = 0
        population_size = self.initial_population_size

        while eval_count < self.budget:
            # Adaptive inertia weight
            w = self.w_final + (self.w_initial - self.w_final) * (self.budget - eval_count) / self.budget
            
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

            # PSO: Update velocities and positions
            for i in range(population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] = (w * self.velocities[i] + 
                                      self.c1 * r1 * (self.personal_best[i] - self.particles[i]) + 
                                      self.c2 * r2 * (self.global_best - self.particles[i]))
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], bounds_lb, bounds_ub)

            # Dynamic DE: Adaptive mutation factor and population resizing
            mutation_factor = 0.8 + 0.2 * np.random.rand() * (1 - eval_count / self.budget)
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
            self.particles[:len(top_indices)] = elites

            # Adjust population size dynamically
            if eval_count < self.budget * 0.5:
                population_size = int(self.initial_population_size * (1 + 0.5 * eval_count / self.budget))
            else:
                population_size = int(self.initial_population_size * (1.5 - 0.5 * eval_count / self.budget))

            if eval_count >= self.budget:
                break

        return self.global_best