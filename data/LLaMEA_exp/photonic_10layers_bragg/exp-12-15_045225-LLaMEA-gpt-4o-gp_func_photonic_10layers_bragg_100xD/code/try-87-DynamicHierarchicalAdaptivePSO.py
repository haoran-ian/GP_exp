import numpy as np

class DynamicHierarchicalAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.initial_c1 = 1.5
        self.initial_c2 = 1.5
        self.initial_w = 0.8
        self.temperature = 100.0
        self.cooling_rate = 0.98
        self.learning_rate = 0.1
        self.quantum_factor = 0.05
        self.hierarchy_factor = 0.1

    def adaptive_population_size(self, evals):
        return max(10, int(self.initial_population_size * (1 - evals / self.budget)))

    def hierarchical_learning(self, particles, personal_best, personal_best_scores, evals):
        if np.random.rand() < self.hierarchy_factor * (1 - evals / self.budget):
            best_indices = np.argsort(personal_best_scores)[:3]
            for i in range(len(particles)):
                particles[i] = np.mean(personal_best[best_indices], axis=0)
        return particles

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        evals = 0
        particles = np.random.uniform(low=lb, high=ub, size=(self.initial_population_size, self.dim))
        velocities = np.random.uniform(size=(self.initial_population_size, self.dim)) * (ub - lb) / 20.0
        personal_best = particles.copy()
        personal_best_scores = np.array([func(p) for p in personal_best])
        global_best_index = np.argmin(personal_best_scores)
        global_best = personal_best[global_best_index]
        global_best_score = personal_best_scores[global_best_index]
        evals += self.initial_population_size

        while evals < self.budget:
            population_size = self.adaptive_population_size(evals)
            particles = self.hierarchical_learning(particles, personal_best, personal_best_scores, evals)
            for i in range(population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.initial_w * velocities[i] +
                                 self.initial_c1 * r1 * (personal_best[i] - particles[i]) +
                                 self.initial_c2 * r2 * (global_best - particles[i]))
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

                if np.random.rand() < self.quantum_factor:
                    particles[i] = np.clip(global_best + np.random.normal(size=self.dim) * (ub - lb) / 10.0, lb, ub)

                score = func(particles[i])
                evals += 1
                if evals >= self.budget:
                    break

                if score < personal_best_scores[i]:
                    personal_best[i] = particles[i]
                    personal_best_scores[i] = score

                if score < global_best_score:
                    global_best = particles[i]
                    global_best_score = score

            self.initial_w = 0.4 + 0.5 * np.cos(np.pi * evals / self.budget)
            self.temperature *= self.cooling_rate

        return global_best