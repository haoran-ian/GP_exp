import numpy as np

class AdaptiveQuantumHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.w_min = 0.3  # minimum inertia weight
        self.w_max = 0.7  # maximum inertia weight
        self.c1 = 1.5     # cognitive coefficient
        self.c2 = 1.5     # social coefficient
        self.F_base = 0.5 # base scaling factor for DE
        self.CR_base = 0.5 # base crossover probability for DE
        self.q_min = -1
        self.q_max = 1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        qubits = np.random.uniform(self.q_min, self.q_max, (self.population_size, self.dim))
        population = lb + (ub - lb) / 2 * (1 + np.tanh(qubits))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        p_best = population.copy()
        p_best_scores = np.array([float('inf')] * self.population_size)

        for i in range(self.population_size):
            score = func(population[i])
            p_best_scores[i] = score
        
        g_best = population[np.argmin(p_best_scores)]
        g_best_score = min(p_best_scores)

        evaluations = self.population_size
        historical_scores = []

        while evaluations < self.budget:
            dynamic_pressure = (self.budget - evaluations) / self.budget
            self.w = self.w_min + (self.w_max - self.w_min) * np.random.rand() * dynamic_pressure
            r1, r2 = np.random.rand(2)
            velocity = (self.w * velocity +
                        self.c1 * r1 * (p_best - population) +
                        self.c2 * r2 * (g_best - population))
            population = np.clip(population + velocity, lb, ub)

            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                self.F = self.F_base + 0.2 * np.random.rand() * dynamic_pressure
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                self.CR = max(self.CR_base + 0.3 * np.random.rand() * (g_best_score / (p_best_scores[i] + 1e-9)), 0.1)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_score = func(trial)
                evaluations += 1
                historical_scores.append(trial_score)
                if trial_score < p_best_scores[i]:
                    p_best[i] = trial
                    p_best_scores[i] = trial_score
                    if trial_score < g_best_score:
                        g_best = trial
                        g_best_score = trial_score
                if evaluations >= self.budget:
                    break

            self.population_size = max(10, int(20 * dynamic_pressure))
            if len(historical_scores) > 5:
                avg_improvement = np.mean(np.diff(historical_scores[-5:]))
                if avg_improvement < 0.001:
                    self.w_max = max(0.5, self.w_max - 0.1)

        return g_best