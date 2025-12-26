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
        w_min = 0.3
        c1, c2 = 1.5, 1.5
        F_base = 0.8
        CR_base = 0.9
        phase_change_threshold = 0.2 * self.budget
        elite_fraction = 0.1  
        elite_count = max(1, int(elite_fraction * population_size))
        
        # Initialize a chaotic sequence using logistic map
        chaotic_sequence = np.zeros(self.budget)
        chaotic_sequence[0] = 0.7  # Initial value
        for i in range(1, self.budget):
            chaotic_sequence[i] = 4 * chaotic_sequence[i-1] * (1 - chaotic_sequence[i-1])
        
        while eval_count < self.budget:
            # Adaptive inertia weight with chaotic sequence
            w = w_max - ((w_max - w_min) * (eval_count**(1.02 + 0.1*chaotic_sequence[eval_count]) / self.budget**1.02))

            # Determine phase based on current budget usage
            if eval_count % (2 * phase_change_threshold) < phase_change_threshold:
                F_adaptive = F_base * (1.3 + 0.1 * chaotic_sequence[eval_count])
                c1_adaptive, c2_adaptive = c1 * (1.1 + 0.1 * chaotic_sequence[eval_count]), c2 * (0.9 - 0.1 * chaotic_sequence[eval_count])
                CR = CR_base + 0.05 * chaotic_sequence[eval_count]
            else:
                F_adaptive = F_base * (0.7 - 0.1 * chaotic_sequence[eval_count])
                c1_adaptive, c2_adaptive = c1 * (0.9 - 0.1 * chaotic_sequence[eval_count]), c2 * (1.1 + 0.1 * chaotic_sequence[eval_count])
                CR = CR_base - 0.05 * chaotic_sequence[eval_count]

            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities + 
                          c1_adaptive * r1 * (personal_best_positions - swarm) + 
                          c2_adaptive * r2 * (global_best_position - swarm))
            swarm += velocities
            swarm = np.clip(swarm, lb, ub)

            # Dynamically adjust elite_fraction
            elite_fraction = max(0.1, 0.1 + 0.1 * (eval_count / self.budget))
            elite_count = max(1, int(elite_fraction * population_size))

            # Differential Evolution mutation and crossover
            for i in range(population_size):
                if i >= elite_count:
                    candidates = list(range(population_size))
                    candidates.remove(i)
                    a, b, c = np.random.choice(candidates, 3, replace=False)

                    mutant = np.clip(swarm[a] + F_adaptive * (swarm[b] - swarm[c]), lb, ub)
                    
                    crossover = np.random.rand(self.dim) < CR
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