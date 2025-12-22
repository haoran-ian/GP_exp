import numpy as np

class EnhancedChaosAdaptiveSwarm:
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
        self.chaotic_factor_initial = 0.5
        self.chaotic_factor_final = 1.2

    def adaptive_parameters(self, eval_ratio):
        c1 = self.c1_initial + eval_ratio * (self.c1_final - self.c1_initial)
        c2 = self.c2_initial + eval_ratio * (self.c2_final - self.c2_initial)
        w = self.w_initial - eval_ratio * (self.w_initial - self.w_final)
        diff_weight = self.diff_weight_initial + eval_ratio * (self.diff_weight_final - self.diff_weight_initial)
        cross_prob = self.cross_prob_initial - eval_ratio * (self.cross_prob_initial - self.cross_prob_final)
        chaotic_factor = self.chaotic_factor_initial + eval_ratio * (self.chaotic_factor_final - self.chaotic_factor_initial)
        return c1, c2, w, diff_weight, cross_prob, chaotic_factor

    def dynamic_leader_selection(self, personal_best_scores):
        leader_indices = np.argsort(personal_best_scores)[:self.leader_selection_pressure]
        return leader_indices

    def logistic_map(self, x):
        return 4 * x * (1 - x)

    def sinusoidal_map(self, x):
        return np.abs(2 * np.sin(np.pi * x))

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
        use_logistic_map = True

        while evaluations < self.budget:
            eval_ratio = evaluations / self.budget
            c1, c2, w, diff_weight, cross_prob, chaotic_factor = self.adaptive_parameters(eval_ratio)

            current_population_size = max(5, int(self.initial_population_size * (1 - eval_ratio)))

            leader_indices = self.dynamic_leader_selection(personal_best_scores)
            new_global_best = personal_best[np.random.choice(leader_indices)]

            for i in range(current_population_size):
                r1, r2 = np.random.rand(2)
                chaos_component = 1.1 * chaotic_factor * (self.logistic_map(chaotic_state) if use_logistic_map else self.sinusoidal_map(chaotic_state))
                velocity[i] = w * velocity[i] + c1 * r1 * (personal_best[i] - swarm[i]) + c2 * r2 * (new_global_best - swarm[i]) + chaos_component
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

            chaotic_state = self.logistic_map(chaotic_state) if use_logistic_map else self.sinusoidal_map(chaotic_state)
            use_logistic_map = not use_logistic_map  # switch map

        return global_best