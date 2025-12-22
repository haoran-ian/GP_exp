import numpy as np

class HybridQuantumSimulatedAnnealingPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.initial_c1 = 2.0
        self.initial_c2 = 1.5
        self.initial_w = 0.9
        self.temperature = 100.0
        self.cooling_rate = 0.95
        self.quantum_factor = 0.1
        self.learning_rate = 0.05

    def adaptive_population_size(self, evals):
        return max(10, int(self.initial_population_size * (1 - evals / self.budget)))

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
            temperature_factor = np.exp(-evals / self.budget) * self.temperature
            for i in range(population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                c1 = self.initial_c1 * (1 - evals / self.budget)
                c2 = self.initial_c2 * (1 + evals / (2 * self.budget))
                velocities[i] = (self.initial_w * velocities[i] +
                                 c1 * r1 * (personal_best[i] - particles[i]) +
                                 c2 * r2 * (global_best - particles[i]))

                if np.random.rand() < self.quantum_factor:
                    quantum_jump = np.random.normal(size=self.dim) * temperature_factor
                    particles[i] = np.clip(global_best + quantum_jump, lb, ub)
                else:
                    particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

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

            self.initial_w = 0.5 + 0.4 * np.cos(np.pi * evals / self.budget)
            self.temperature *= self.cooling_rate

        return global_best