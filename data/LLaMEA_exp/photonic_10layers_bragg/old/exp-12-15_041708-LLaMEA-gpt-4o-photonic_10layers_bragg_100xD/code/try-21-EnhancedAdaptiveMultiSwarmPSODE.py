import numpy as np

class EnhancedAdaptiveMultiSwarmPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = 50
        num_swarms = 3
        swarm_size = population_size // num_swarms
        swarms = [np.random.uniform(lb, ub, (swarm_size, self.dim)) for _ in range(num_swarms)]
        velocities = [np.random.uniform(-1, 1, (swarm_size, self.dim)) for _ in range(num_swarms)]
        personal_best_positions = [np.copy(swarm) for swarm in swarms]
        personal_best_scores = [np.array([func(ind) for ind in swarm]) for swarm in swarms]
        global_best_positions = [np.copy(personal_best_positions[i][np.argmin(personal_best_scores[i])]) for i in range(num_swarms)]
        global_best_scores = [np.min(personal_best_scores[i]) for i in range(num_swarms)]
        
        eval_count = population_size

        # Adaptive parameters
        w_max = 0.9
        w_min = 0.4
        c1, c2 = 1.5, 1.5
        F_base = 0.8
        CR = 0.9
        phase_change_threshold = 0.2 * self.budget
        elite_fraction = 0.1
        elite_count = max(1, int(elite_fraction * swarm_size))
        
        while eval_count < self.budget:
            # Adaptive inertia weight
            w = w_max - ((w_max - w_min) * (eval_count / self.budget))

            # Determine phase based on current budget usage
            if eval_count % (2 * phase_change_threshold) < phase_change_threshold:
                F_adaptive = F_base * 1.3
                c1_adaptive, c2_adaptive = c1 * 1.1, c2 * 0.9
            else:
                F_adaptive = F_base * 0.7
                c1_adaptive, c2_adaptive = c1 * 0.9, c2 * 1.1

            # Multi-swarm cooperation and local search
            for i in range(num_swarms):
                # Particle Swarm Optimization update
                r1, r2 = np.random.rand(2)
                velocities[i] = (w * velocities[i] + 
                                 c1_adaptive * r1 * (personal_best_positions[i] - swarms[i]) + 
                                 c2_adaptive * r2 * (global_best_positions[i] - swarms[i]))
                swarms[i] += velocities[i]
                swarms[i] = np.clip(swarms[i], lb, ub)

                # Differential Evolution mutation and crossover
                for j in range(swarm_size):
                    if j >= elite_count:
                        candidates = list(range(swarm_size))
                        candidates.remove(j)
                        a, b, c = np.random.choice(candidates, 3, replace=False)

                        mutant = np.clip(swarms[i][a] + F_adaptive * (swarms[i][b] - swarms[i][c]), lb, ub)
                        crossover = np.random.rand(self.dim) < CR
                        trial = np.where(crossover, mutant, swarms[i][j])

                        f_trial = func(trial)
                        eval_count += 1

                        if f_trial < personal_best_scores[i][j]:
                            personal_best_positions[i][j] = trial
                            personal_best_scores[i][j] = f_trial
                            if f_trial < global_best_scores[i]:
                                global_best_positions[i] = trial
                                global_best_scores[i] = f_trial

                        if eval_count >= self.budget:
                            break

                # Multi-swarm communication
                if np.random.rand() < 0.3:
                    for j in range(num_swarms):
                        if global_best_scores[j] < global_best_scores[i]:
                            global_best_positions[i] = global_best_positions[j]
                            global_best_scores[i] = global_best_scores[j]

        # Find the best solution across all swarms
        best_swarm_index = np.argmin(global_best_scores)
        return {'best_position': global_best_positions[best_swarm_index], 'best_score': global_best_scores[best_swarm_index]}