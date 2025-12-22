import numpy as np

class RefinedAdaptiveLearningPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.c1_start = 2.5    # Increased cognitive coefficient for better exploration
        self.c2_start = 0.5    # Lowered initial social coefficient to reduce premature convergence
        self.w_start = 0.9     # Initial inertia weight
        self.cooling_rate = 0.99 # Enhanced cooling rate for consistent global exploration
        self.learning_rate = 0.15 # Higher learning rate for faster contextual learning
        self.diversification_rate = 0.1 # Additional diversification rate for avoiding local optima

    def adaptive_population_size(self, evals):
        return max(5, int(self.initial_population_size * (1 - evals / self.budget)))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        evals = 0
        particles = np.random.uniform(low=lb, high=ub, size=(self.initial_population_size, self.dim))
        velocities = np.random.uniform(size=(self.initial_population_size, self.dim)) * (ub - lb) / 50.0
        personal_best = particles.copy()
        personal_best_scores = np.array([func(p) for p in personal_best])
        global_best_index = np.argmin(personal_best_scores)
        global_best = personal_best[global_best_index]
        global_best_score = personal_best_scores[global_best_index]
        evals += self.initial_population_size

        while evals < self.budget:
            population_size = self.adaptive_population_size(evals)
            for i in range(population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                c1 = self.c1_start * (1 - evals / self.budget)
                c2 = self.c2_start + (2.0 - self.c2_start) * (evals / self.budget)
                velocity_scaling = 0.7 + 0.3 * np.random.rand() * (1 - evals / self.budget)  # Enhanced adaptive velocity scaling
                velocities[i] = (self.w_start * velocities[i] +
                                 c1 * r1 * (personal_best[i] - particles[i]) +
                                 c2 * r2 * (global_best - particles[i])) * velocity_scaling
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

                # Evaluate new position
                score = func(particles[i])
                evals += 1
                if evals >= self.budget:
                    break

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best[i] = particles[i]
                    personal_best_scores[i] = score

                # Contextual learning: learn from global best when stuck
                if score > personal_best_scores[i] and np.random.rand() < self.learning_rate:
                    particles[i] = np.clip(global_best + np.random.randn(self.dim) * np.abs(global_best - particles[i]), lb, ub)
                    score = func(particles[i])
                    evals += 1

                # Diversification strategy
                if np.random.rand() < self.diversification_rate:
                    particles[i] = np.random.uniform(low=lb, high=ub, size=self.dim)

                # Update global best
                if score < global_best_score:
                    global_best = particles[i]
                    global_best_score = score

            self.w_start = 0.4 + 0.5 * (1 - evals / self.budget)  # Adaptive inertia weight
            self.cooling_rate *= 0.98  # Gradual cooling for more refined search

        return global_best