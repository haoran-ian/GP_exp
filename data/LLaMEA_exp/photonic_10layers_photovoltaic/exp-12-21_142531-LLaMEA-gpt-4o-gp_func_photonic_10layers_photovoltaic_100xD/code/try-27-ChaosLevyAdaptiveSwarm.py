import numpy as np

class ChaosLevyAdaptiveSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30
        self.c1_initial = 1.5
        self.c2_initial = 1.5
        self.w_initial = 0.9
        self.diff_weight_initial = 0.5
        self.cross_prob_initial = 0.9
        self.c1_final = 2.5
        self.c2_final = 0.5
        self.w_final = 0.4
        self.diff_weight_final = 1.0
        self.cross_prob_final = 0.8
        self.leader_selection_pressure = 5
        self.chaotic_factor = 0.7
        self.levy_beta = 1.5

    def adaptive_parameters(self, eval_ratio):
        c1 = self.c1_initial + eval_ratio * (self.c1_final - self.c1_initial)
        c2 = self.c2_initial + eval_ratio * (self.c2_final - self.c2_initial)
        w = self.w_initial - eval_ratio * (self.w_initial - self.w_final)
        diff_weight = self.diff_weight_initial + eval_ratio * (self.diff_weight_final - self.diff_weight_initial)
        cross_prob = self.cross_prob_initial - eval_ratio * (self.cross_prob_initial - self.cross_prob_final)
        return c1, c2, w, diff_weight, cross_prob

    def dynamic_leader_selection(self, personal_best_scores):
        leader_indices = np.argsort(personal_best_scores)[:self.leader_selection_pressure]
        return leader_indices

    def chaotic_map(self, x):
        return 4 * x * (1 - x)  # logistic map for chaos

    def levy_flight(self):
        sigma = (np.math.gamma(1 + self.levy_beta) * np.sin(np.pi * self.levy_beta / 2) /
                 (np.math.gamma((1 + self.levy_beta) / 2) * self.levy_beta * 2**((self.levy_beta - 1) / 2)))**(1 / self.levy_beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        return u / np.abs(v)**(1 / self.levy_beta)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.initial_population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.initial_population_size, self.dim))
        personal_best = np.copy(swarm)
        personal_best_scores = np.array([func(p) for p in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)
        
        evaluations = self.initial_population_size
        chaotic_state = np.random.rand()

        while evaluations < self.budget:
            eval_ratio = evaluations / self.budget
            c1, c2, w, diff_weight, cross_prob = self.adaptive_parameters(eval_ratio)

            current_population_size = max(5, int(self.initial_population_size * (1 - eval_ratio)))

            leader_indices = self.dynamic_leader_selection(personal_best_scores)
            new_global_best = personal_best[np.random.choice(leader_indices)]

            for i in range(current_population_size):
                r1, r2 = np.random.rand(2)
                chaos_component = self.chaotic_factor * self.chaotic_map(chaotic_state)
                levy_component = self.levy_flight()
                velocity[i] = 0.5 * (
                    w * velocity[i] +
                    c1 * r1 * (personal_best[i] - swarm[i]) +
                    c2 * r2 * (new_global_best - swarm[i]) +
                    chaos_component + levy_component
                )
                swarm[i] = np.clip(swarm[i] + velocity[i], lb, ub)
                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_scores[i]:
                    personal_best_scores[i] = f_value
                    personal_best[i] = swarm[i]
                    if f_value < global_best_score:
                        global_best_score = f_value
                        global_best = swarm[i]

            for i in range(current_population_size):
                if evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(current_population_size) if idx != i]
                a, b, c = swarm[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + diff_weight * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < cross_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, swarm[i])
                f_trial = func(trial)
                evaluations += 1
                if f_trial < personal_best_scores[i]:
                    personal_best_scores[i] = f_trial
                    personal_best[i] = trial
                    if f_trial < global_best_score:
                        global_best_score = f_trial
                        global_best = trial

            chaotic_state = self.chaotic_map(chaotic_state)

        return global_best