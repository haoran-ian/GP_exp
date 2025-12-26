import numpy as np

class EnhancedAdaptiveQuantumHybridPSO_DE_DNI:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.w_min = 0.3
        self.w_max = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.F_base = 0.5
        self.CR_base = 0.5
        self.q_min = -1
        self.q_max = 1
        self.momentum_factor = 0.9  # Momentum term
        self.local_influence = 0.1  # New local neighborhood influence factor

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
        historical_improvements = []

        while evaluations < self.budget:
            dynamic_pressure = (self.budget - evaluations) / self.budget
            self.w = self.w_min + (self.w_max - self.w_min) * np.random.rand() * dynamic_pressure
            r1, r2 = np.random.rand(2)
            velocity = (self.momentum_factor * velocity +
                        self.w * (self.c1 * r1 * (p_best - population) +
                                  self.c2 * r2 * (g_best - population)))
            
            # Integrating local neighborhood information
            for i in range(self.population_size):
                local_best_score = float('inf')
                local_best = population[i]
                for j in range(self.population_size):
                    if i != j:
                        distance = np.linalg.norm(population[i] - population[j])
                        if distance < (ub - lb) / 4 and p_best_scores[j] < local_best_score:
                            local_best_score = p_best_scores[j]
                            local_best = population[j]
                velocity[i] += self.local_influence * (local_best - population[i])
                
            population = np.clip(population + velocity, lb, ub)

            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                avg_population_score = np.mean(p_best_scores)
                self.F = self.F_base + 0.4 * np.exp(-abs(g_best_score - avg_population_score)) * np.random.rand()
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
                    historical_improvements.append(p_best_scores[i] - trial_score)
                    if trial_score < g_best_score:
                        g_best = trial
                        g_best_score = trial_score
                if evaluations >= self.budget:
                    break

            if len(historical_scores) > 5:
                avg_improvement = np.mean(np.diff(historical_scores[-5:]))
                if avg_improvement < 0.001:
                    self.w_max = max(0.5, self.w_max - 0.1)

            if len(historical_improvements) > 5:
                momentum_improvement = np.mean(historical_improvements[-5:])
                if momentum_improvement < 0.001:
                    self.momentum_factor = max(0.7, self.momentum_factor - 0.05)

        return g_best