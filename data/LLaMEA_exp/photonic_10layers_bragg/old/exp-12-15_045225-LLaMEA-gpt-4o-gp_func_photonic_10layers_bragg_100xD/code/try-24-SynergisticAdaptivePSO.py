import numpy as np

class SynergisticAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.init_population_size = 50
        self.c1 = 2.0
        self.c2 = 1.0
        self.w = 0.9
        self.cooling_rate = 0.98
        self.learning_rate = 0.1
        self.diversity_threshold = 0.001

    def adaptive_population_size(self, evals):
        return max(5, int(self.init_population_size * (1 - evals / self.budget)))

    def calculate_diversity(self, particles):
        return np.mean(np.std(particles, axis=0))

    def dynamically_adjust_neighborhood(self, particles, global_best, evals):
        diversity = self.calculate_diversity(particles)
        if diversity < self.diversity_threshold:
            # Widen the neighborhood if diversity is low
            return global_best + np.random.randn(self.dim) * np.mean(np.abs(particles - global_best), axis=0)
        else:
            # Otherwise, refine around the current global best
            return global_best

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        evals = 0
        particles = np.random.uniform(low=lb, high=ub, size=(self.init_population_size, self.dim))
        velocities = np.random.uniform(size=(self.init_population_size, self.dim)) * (ub - lb) / 20.0
        personal_best = particles.copy()
        personal_best_scores = np.array([func(p) for p in personal_best])
        global_best_index = np.argmin(personal_best_scores)
        global_best = personal_best[global_best_index]
        global_best_score = personal_best_scores[global_best_index]
        evals += self.init_population_size

        while evals < self.budget:
            population_size = self.adaptive_population_size(evals)
            for i in range(population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                c1 = self.c1 * (1 - evals / self.budget)
                c2 = self.c2 + (2.0 - self.c2) * (evals / self.budget)
                velocity_scaling = 0.4 + 0.6 * np.random.rand() * (1 - evals / self.budget)
                velocities[i] = (self.w * velocities[i] +
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

                # Contextual learning: adaptive neighborhood strategy
                if score > personal_best_scores[i] and np.random.rand() < self.learning_rate:
                    particles[i] = np.clip(self.dynamically_adjust_neighborhood(particles, global_best, evals), lb, ub)
                    score = func(particles[i])
                    evals += 1

                # Update global best
                if score < global_best_score:
                    global_best = particles[i]
                    global_best_score = score

            self.w = 0.4 + 0.5 * (1 - evals / self.budget)
            self.cooling_rate *= self.cooling_rate

        return global_best