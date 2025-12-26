import numpy as np

class QuantumInspiredDynamicPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.w_min = 0.3
        self.w_max = 0.9
        self.c1_base = 1.5
        self.c2_base = 1.5
        self.q_min = -1
        self.q_max = 1
        self.momentum_factor = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        qubits = np.random.uniform(self.q_min, self.q_max, (self.population_size, self.dim))
        population = lb + (ub - lb) / 2 * (1 + np.tanh(qubits))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        p_best = population.copy()
        p_best_scores = np.array([func(ind) for ind in population])

        g_best = population[np.argmin(p_best_scores)]
        g_best_score = np.min(p_best_scores)
        evaluations = self.population_size

        while evaluations < self.budget:
            diversity = np.std(population)
            c1 = self.c1_base + 0.5 * (1 - diversity)
            c2 = self.c2_base + 0.5 * diversity

            dynamic_pressure = (self.budget - evaluations) / self.budget
            w = self.w_min + (self.w_max - self.w_min) * np.random.rand() * dynamic_pressure

            r1, r2 = np.random.rand(2)
            velocity = (self.momentum_factor * velocity +
                        w * (c1 * r1 * (p_best - population) +
                             c2 * r2 * (g_best - population)))
            population = np.clip(population + velocity, lb, ub)

            for i in range(self.population_size):
                trial_score = func(population[i])
                evaluations += 1
                if trial_score < p_best_scores[i]:
                    p_best[i] = population[i]
                    p_best_scores[i] = trial_score
                    if trial_score < g_best_score:
                        g_best = population[i]
                        g_best_score = trial_score
                if evaluations >= self.budget:
                    break

        return g_best