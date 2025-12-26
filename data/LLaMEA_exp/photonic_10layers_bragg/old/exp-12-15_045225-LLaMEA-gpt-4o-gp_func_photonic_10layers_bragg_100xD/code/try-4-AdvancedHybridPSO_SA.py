import numpy as np

class AdvancedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.initial_c1 = 2.5  # starting cognitive coefficient
        self.initial_c2 = 0.5  # starting social coefficient
        self.w = 0.5   # inertia weight
        self.temperature = 100.0  # initial temperature for simulated annealing
        self.cooling_rate = 0.95  # enhanced cooling rate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(low=lb, high=ub, size=(self.population_size, self.dim))
        velocities = np.random.uniform(size=(self.population_size, self.dim)) * (ub - lb) / 20.0
        personal_best = particles.copy()
        personal_best_scores = np.array([func(p) for p in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)
        evals = self.population_size

        while evals < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                c1 = self.initial_c1 * (1 - evals / self.budget)
                c2 = self.initial_c2 + (2.0 - self.initial_c2) * (evals / self.budget)
                velocities[i] = (self.w * velocities[i] +
                                 c1 * r1 * (personal_best[i] - particles[i]) +
                                 c2 * r2 * (global_best - particles[i]))
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

                # Apply enhanced simulated annealing for local refinement
                delta_score = score - personal_best_scores[i]
                acceptance_prob = np.exp(-delta_score / self.temperature) if delta_score > 0 else 1.0
                if np.random.rand() < acceptance_prob:
                    personal_best[i] = particles[i]
                    personal_best_scores[i] = score

                # Update global best
                if score < global_best_score:
                    global_best = particles[i]
                    global_best_score = score

            self.w = 0.4 + 0.5 * (1 - evals / self.budget)  # Adaptive inertia weight
            self.temperature *= self.cooling_rate  # Cooling down the temperature

        return global_best