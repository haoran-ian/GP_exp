import numpy as np

class MultiSwarmAdaptiveLearningPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.initial_c1 = 2.0
        self.initial_c2 = 1.0
        self.initial_w = 0.9
        self.temperature = 100.0
        self.cooling_rate = 0.98
        self.learning_rate = 0.1
        self.num_swarms = 3  # Introducing multiple swarms

    def adaptive_population_size(self, evals):
        return max(5, int(self.initial_population_size * (1 - evals / self.budget)))

    def gradient_approximation(self, particles, scores):
        # Simple gradient approximation using finite differences
        gradients = np.zeros_like(particles)
        h = 1e-5
        for i in range(particles.shape[0]):
            for j in range(self.dim):
                x_plus_h = particles[i].copy()
                x_plus_h[j] += h
                gradients[i, j] = (func(x_plus_h) - scores[i]) / h
        return gradients

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        evals = 0
        populations = [np.random.uniform(low=lb, high=ub, size=(self.initial_population_size, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(size=(self.initial_population_size, self.dim)) * (ub - lb) / 20.0 for _ in range(self.num_swarms)]
        personal_best = [pop.copy() for pop in populations]
        personal_best_scores = [np.array([func(p) for p in pop]) for pop in personal_best]
        global_best_index = [np.argmin(scores) for scores in personal_best_scores]
        global_best = [personal_best[swarm][index] for swarm, index in enumerate(global_best_index)]
        global_best_score = [scores[index] for scores, index in zip(personal_best_scores, global_best_index)]
        evals += self.initial_population_size * self.num_swarms

        while evals < self.budget:
            for swarm in range(self.num_swarms):
                population_size = self.adaptive_population_size(evals)
                for i in range(population_size):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    c1 = self.initial_c1 * (1 - evals / self.budget)
                    c2 = self.initial_c2 + (2.0 - self.initial_c2) * (evals / self.budget)
                    velocity_scaling = 0.4 + 0.6 * np.random.rand() * (1 - evals / self.budget)
                    velocities[swarm][i] = (self.initial_w * velocities[swarm][i] +
                                            c1 * r1 * (personal_best[swarm][i] - populations[swarm][i]) +
                                            c2 * r2 * (global_best[swarm] - populations[swarm][i])) * velocity_scaling
                    populations[swarm][i] = np.clip(populations[swarm][i] + velocities[swarm][i], lb, ub)

                    # Evaluate new position
                    score = func(populations[swarm][i])
                    evals += 1
                    if evals >= self.budget:
                        break

                    # Update personal best
                    if score < personal_best_scores[swarm][i]:
                        personal_best[swarm][i] = populations[swarm][i]
                        personal_best_scores[swarm][i] = score

                    # Gradient-inspired learning: adjust position using gradient approximation
                    if np.random.rand() < self.learning_rate:
                        gradients = self.gradient_approximation(populations[swarm], personal_best_scores[swarm])
                        populations[swarm][i] -= self.learning_rate * gradients[i]
                        populations[swarm][i] = np.clip(populations[swarm][i], lb, ub)
                        score = func(populations[swarm][i])
                        evals += 1

                    # Update global best
                    if score < global_best_score[swarm]:
                        global_best[swarm] = populations[swarm][i]
                        global_best_score[swarm] = score

            self.initial_w = 0.4 + 0.5 * (1 - evals / self.budget)
            self.temperature *= self.cooling_rate

        # Select the best solution across all swarms
        best_swarm_index = np.argmin(global_best_score)
        return global_best[best_swarm_index]