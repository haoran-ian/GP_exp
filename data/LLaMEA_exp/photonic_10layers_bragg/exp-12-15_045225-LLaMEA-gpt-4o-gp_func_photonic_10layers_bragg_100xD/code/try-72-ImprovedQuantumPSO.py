import numpy as np

class ImprovedQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.quantum_factor = 0.05
        self.learning_rate = 0.1

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
            w = self.w_max - (self.w_max - self.w_min) * (evals / self.budget)
            for i in range(self.initial_population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - particles[i]) +
                                 self.c2 * r2 * (global_best - particles[i]))
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

                if np.random.rand() < self.quantum_factor:
                    particles[i] = np.clip(global_best + np.random.normal(size=self.dim) * (ub - lb) / 20.0, lb, ub)

                score = func(particles[i])
                evals += 1
                if evals >= self.budget:
                    break

                if score < personal_best_scores[i]:
                    personal_best[i] = particles[i]
                    personal_best_scores[i] = score

                if score > personal_best_scores[i] and np.random.rand() < self.learning_rate:
                    new_neighbors = np.random.choice(range(self.initial_population_size), 3, replace=False)
                    best_neighbor = min(new_neighbors, key=lambda n: personal_best_scores[n])
                    particles[i] = np.clip(personal_best[best_neighbor] + np.random.randn(self.dim) * 0.5 * np.abs(personal_best[best_neighbor] - particles[i]), lb, ub)
                    score = func(particles[i])
                    evals += 1

                if score < global_best_score:
                    global_best = particles[i]
                    global_best_score = score

        return global_best