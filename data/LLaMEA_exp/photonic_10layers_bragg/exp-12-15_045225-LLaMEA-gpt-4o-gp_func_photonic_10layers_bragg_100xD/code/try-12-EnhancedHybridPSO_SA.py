import numpy as np

class EnhancedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.initial_c1 = 2.5  # starting cognitive coefficient
        self.initial_c2 = 0.5  # starting social coefficient
        self.initial_w = 0.9   # initial inertia weight
        self.temperature = 100.0  # initial temperature for simulated annealing
        self.cooling_rate = 0.93  # refined cooling rate
        self.quantum_tunneling_factor = 0.15  # enhanced factor for quantum tunneling

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
            for i in range(population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                c1 = self.initial_c1 * (1 - evals / self.budget)
                c2 = self.initial_c2 + (2.0 - self.initial_c2) * (evals / self.budget)
                velocities[i] = (self.initial_w * velocities[i] +
                                 c1 * r1 * (personal_best[i] - particles[i]) +
                                 c2 * r2 * (global_best - particles[i]))
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

                # Evaluate new position
                score = func(particles[i])
                evals += 1
                if evals >= self.budget:
                    break

                # Update personal best if better
                if score < personal_best_scores[i]:
                    personal_best[i] = particles[i]
                    personal_best_scores[i] = score

                # Apply enhanced simulated annealing with quantum-inspired tunneling
                delta_score = score - personal_best_scores[i]
                tunneling_prob = np.exp(-self.quantum_tunneling_factor * delta_score) if delta_score > 0 else 1.0
                if np.random.rand() < tunneling_prob:
                    personal_best[i] = particles[i]
                    personal_best_scores[i] = score

                # Update global best
                if score < global_best_score:
                    global_best = particles[i]
                    global_best_score = score

            self.initial_w = 0.4 + 0.5 * (1 - evals / self.budget)  # Adaptive inertia weight
            self.temperature *= self.cooling_rate  # Cooling down the temperature

        return global_best