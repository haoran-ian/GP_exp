import numpy as np

class EnhancedAdaptiveMultiPhasePSODE:
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

        # Adaptive parameters
        w_max = 0.9
        w_min = 0.4
        c1, c2 = 1.5, 1.5
        F_base = 0.8
        CR_base = 0.9
        phase_change_threshold = 0.3 * self.budget

        def diversity_metric(swarm):
            return np.mean(np.std(swarm, axis=0))

        while eval_count < self.budget:
            # Adaptive inertia weight
            w = w_max - ((w_max - w_min) * (eval_count / self.budget))

            current_diversity = diversity_metric(swarm)
            
            # Determine phase based on current budget usage
            if eval_count < phase_change_threshold:
                # Exploration phase: prioritize exploration
                F_adaptive = F_base * (1.1 + 0.5 * current_diversity)
                c1_adaptive, c2_adaptive = c1 * 1.2, c2 * 0.8
                CR_adaptive = CR_base * (1.2 * current_diversity)
            else:
                # Exploitation phase: prioritize convergence
                F_adaptive = F_base * (0.8 + 0.3 * current_diversity)
                c1_adaptive, c2_adaptive = c1 * 0.8, c2 * 1.2
                CR_adaptive = CR_base * (0.8 * current_diversity)

            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities + 
                          c1_adaptive * r1 * (personal_best_positions - swarm) + 
                          c2_adaptive * r2 * (global_best_position - swarm))
            swarm += velocities
            swarm = np.clip(swarm, lb, ub)

            # Differential Evolution mutation and crossover
            for i in range(population_size):
                candidates = list(range(population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)

                mutant = np.clip(swarm[a] + F_adaptive * (swarm[b] - swarm[c]), lb, ub)
                
                crossover = np.random.rand(self.dim) < CR_adaptive
                trial = np.where(crossover, mutant, swarm[i])

                f_trial = func(trial)
                eval_count += 1

                if f_trial < personal_best_scores[i]:
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = f_trial
                    if f_trial < global_best_score:
                        global_best_position = trial
                        global_best_score = f_trial

                if eval_count >= self.budget:
                    break

        return {'best_position': global_best_position, 'best_score': global_best_score}