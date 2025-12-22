import numpy as np

class AdaptiveMemeticPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = 50
        swarm = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(ind) for ind in swarm])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = np.copy(personal_best_positions[global_best_index])
        global_best_score = personal_best_scores[global_best_index]

        eval_count = population_size

        # Adaptive PSO parameters
        w_max = 0.9
        w_min = 0.4
        c1_min, c1_max = 1.5, 2.5
        c2_min, c2_max = 1.5, 2.5
        local_search_prob = 0.1

        while eval_count < self.budget:
            # Adaptive inertia weight and acceleration coefficients
            w = w_max - ((w_max - w_min) * (eval_count / self.budget))
            c1 = c1_min + ((c1_max - c1_min) * (eval_count / self.budget))
            c2 = c2_max - ((c2_max - c2_min) * (eval_count / self.budget))

            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(2, population_size, self.dim)
            velocities = (w * velocities +
                          c1 * r1 * (personal_best_positions - swarm) +
                          c2 * r2 * (global_best_position - swarm))
            swarm += velocities
            swarm = np.clip(swarm, lb, ub)

            # Evaluate new swarm positions
            for i in range(population_size):
                f_swarm = func(swarm[i])
                eval_count += 1

                if f_swarm < personal_best_scores[i]:
                    personal_best_positions[i] = swarm[i]
                    personal_best_scores[i] = f_swarm
                    if f_swarm < global_best_score:
                        global_best_position = swarm[i]
                        global_best_score = f_swarm

                # Local search for exploitation
                if np.random.rand() < local_search_prob:
                    local_best = self.local_search(func, swarm[i], lb, ub)
                    f_local_best = func(local_best)
                    eval_count += 1
                    if f_local_best < personal_best_scores[i]:
                        personal_best_positions[i] = local_best
                        personal_best_scores[i] = f_local_best
                        if f_local_best < global_best_score:
                            global_best_position = local_best
                            global_best_score = f_local_best

                if eval_count >= self.budget:
                    break

        return {'best_position': global_best_position, 'best_score': global_best_score}

    def local_search(self, func, position, lb, ub):
        # Simple random walk as local search
        step_size = 0.1
        candidate = position + step_size * np.random.uniform(-1, 1, self.dim)
        candidate = np.clip(candidate, lb, ub)
        return candidate