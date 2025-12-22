import numpy as np

class EnhancedParticleDELevyOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.population_size = self.initial_population_size
        self.w_init, self.w_final = 0.7, 0.4
        self.c1_init, self.c1_final = 2.5, 1.5
        self.c2_init, self.c2_final = 2.0, 2.5
        self.F = 0.8
        self.CR = 0.9
        self.alpha_init, self.alpha_final = 0.01, 0.02

    def adapt_parameters(self, iteration, max_iterations):
        progress = iteration / max_iterations
        self.w = self.w_init - (self.w_init - self.w_final) * progress
        self.c1 = self.c1_init - (self.c1_init - self.c1_final) * progress
        self.c2 = self.c2_init + (self.c2_final - self.c2_init) * progress
        self.alpha = self.alpha_init + (self.alpha_final - self.alpha_init) * progress

    def levy_flight(self, scale):
        u = np.random.normal(0, 1, self.dim) * scale
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 3))
        return step

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)

        particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) / 5
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size
        max_iterations = self.budget // self.population_size
        iteration = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (global_best_position - particles[i]))

                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = particles[indices]
                mutant_vector = x1 + self.F * (x2 - x3)
                crossover = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover, mutant_vector, particles[i])

                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
                trial_vector = np.clip(trial_vector, lb, ub)

                if np.random.rand() < 0.3:
                    trial_vector += self.levy_flight(self.alpha)

                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score

                    if trial_score < global_best_score:
                        global_best_position = trial_vector
                        global_best_score = trial_score

                if evaluations >= self.budget:
                    break

            if iteration % (max_iterations // 3) == 0:
                if evaluations < (2 * self.budget) / 3:
                    self.population_size = max(10, self.initial_population_size // 2)
                else:
                    self.population_size = max(5, self.initial_population_size // 4)

            self.adapt_parameters(iteration, max_iterations)
            iteration += 1

        return global_best_position, global_best_score