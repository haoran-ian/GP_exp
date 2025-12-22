import numpy as np

class EnhancedAdaptiveCMA_PSO_DE:
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
        w_max, w_min = 0.9, 0.2
        c1, c2 = 2.05, 2.05
        F_base = 0.5
        CR_base = 0.9

        elite_fraction = 0.1
        elite_count = max(1, int(elite_fraction * population_size))

        # CMA parameters
        cov_matrix = np.eye(self.dim)
        
        while eval_count < self.budget:
            # Adaptive inertia weight and phase balance
            w = w_max - ((w_max - w_min) * eval_count / self.budget)
            phase = eval_count / self.budget
            if phase < 0.5:
                F_adaptive = F_base * (1 + phase)
                CR = CR_base - 0.1 * phase
            else:
                F_adaptive = F_base * (2 - phase)
                CR = CR_base + 0.1 * (phase - 0.5)

            # PSO update strategy
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities + 
                          c1 * r1 * (personal_best_positions - swarm) + 
                          c2 * r2 * (global_best_position - swarm))
            swarm += velocities
            swarm = np.clip(swarm, lb, ub)

            # Elite-based DE mutation and crossover
            for i in range(population_size):
                if i >= elite_count:
                    candidates = list(range(population_size))
                    candidates.remove(i)

                    # CMA-based mutation
                    elite_positions = swarm[:elite_count]
                    elite_mean = np.mean(elite_positions, axis=0)
                    z = np.random.multivariate_normal(np.zeros(self.dim), cov_matrix)
                    mutant = elite_mean + F_adaptive * z
                    mutant = np.clip(mutant, lb, ub)

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

            # Update covariance matrix
            cov_matrix = np.cov(swarm.T)

        return {'best_position': global_best_position, 'best_score': global_best_score}