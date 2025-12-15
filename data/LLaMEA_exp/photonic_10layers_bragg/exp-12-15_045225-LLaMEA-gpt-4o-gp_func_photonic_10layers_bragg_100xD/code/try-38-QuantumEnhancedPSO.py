import numpy as np

class QuantumEnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.initial_c1 = 2.0
        self.initial_c2 = 2.0
        self.initial_w = 0.9
        self.temperature = 100.0
        self.cooling_rate = 0.98
        self.learning_rate = 0.1
        self.quantum_factor = 0.05

    def adaptive_population_size(self, evals):
        return max(5, int(self.initial_population_size * (1 - evals / self.budget)))

    def chaotic_initialization(self, size, lb, ub):
        # Use logistic map for chaotic initialization
        x = np.random.rand(size, self.dim)
        chaotic_sequence = 4 * x * (1 - x)  # Logistic map
        return lb + (ub - lb) * chaotic_sequence

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        evals = 0
        particles = self.chaotic_initialization(self.initial_population_size, lb, ub)
        velocities = np.random.uniform(size=(self.initial_population_size, self.dim)) * (ub - lb) / 20.0
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
                c1 = self.initial_c1 * (1 - evals / self.budget)
                c2 = self.initial_c2 * (1 - evals / self.budget) + (2.0 - self.initial_c2) * (evals / self.budget)
                velocity_scaling = 0.4 + 0.6 * np.random.rand()
                velocities[i] = (self.initial_w * velocities[i] +
                                 c1 * r1 * (personal_best[i] - particles[i]) +
                                 c2 * r2 * (global_best - particles[i])) * velocity_scaling
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

                adaptive_qf = self.quantum_factor * (1 - evals / self.budget)
                if np.random.rand() < adaptive_qf:
                    particles[i] = np.clip(global_best + np.random.normal(size=self.dim) * (ub - lb) / 10.0, lb, ub)

                score = func(particles[i])
                evals += 1
                if evals >= self.budget:
                    break

                if score < personal_best_scores[i]:
                    personal_best[i] = particles[i]
                    personal_best_scores[i] = score

                if np.random.rand() < self.learning_rate * (1 - evals / self.budget):
                    rand_indices = np.random.choice(range(population_size), 3, replace=False)
                    a, b, c = personal_best[rand_indices]
                    trial_vector = np.clip(a + 0.5 * (b - c), lb, ub)
                    trial_score = func(trial_vector)
                    evals += 1
                    if trial_score < personal_best_scores[i]:
                        personal_best[i] = trial_vector
                        personal_best_scores[i] = trial_score

                if score < global_best_score:
                    global_best = particles[i]
                    global_best_score = score

            self.initial_w = 0.5 + 0.4 * (1 - evals / self.budget)
            self.temperature *= self.cooling_rate

        return global_best