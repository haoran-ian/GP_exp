import numpy as np

class DynamicNichingCooperativeSwarmOptimization:
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
        w_max, w_min = 0.9, 0.4
        c1, c2 = 2.0, 2.0
        F_base = 0.7
        CR = 0.9
        adaptive_phase_threshold = 0.2 * self.budget
        
        elite_fraction = 0.1
        elite_count = max(1, int(elite_fraction * population_size))

        # Covariance Matrix Adaptation parameters
        cov_matrix = np.eye(self.dim)
        mean = np.mean(swarm, axis=0)
        
        while eval_count < self.budget:
            # Dynamically adjust inertia weight and learning coefficients
            w = w_max - ((w_max - w_min) * (eval_count / self.budget))
            c1_adaptive = c1 * (1 - (eval_count / self.budget)**2)
            c2_adaptive = c2 * ((eval_count / self.budget)**2)

            # Determine phase dynamically based on performance
            if eval_count % (2 * adaptive_phase_threshold) < adaptive_phase_threshold:
                F_adaptive = F_base * 1.2
                CR = 0.95
            else:
                F_adaptive = F_base * 0.6
                CR = 0.85

            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities + 
                          c1_adaptive * r1 * (personal_best_positions - swarm) + 
                          c2_adaptive * r2 * (global_best_position - swarm))
            swarm += velocities
            swarm = np.clip(swarm, lb, ub)

            # Elite niching with cooperative strategy
            elite_fraction = max(0.1, 0.1 + 0.2 * (eval_count / self.budget))
            elite_count = max(1, int(elite_fraction * population_size))

            # Differential Evolution mutation and crossover with enhanced recombination
            for i in range(population_size):
                if i >= elite_count:
                    candidates = list(range(population_size))
                    candidates.remove(i)

                    # Combined mutation using elite and global influence
                    elite_positions = swarm[:elite_count]
                    elite_mean = np.mean(elite_positions, axis=0)
                    global_influence = global_best_position + np.random.multivariate_normal(np.zeros(self.dim), cov_matrix)
                    mutant = (elite_mean + global_influence) / 2
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

            # Update mean and covariance matrix
            mean = np.mean(swarm, axis=0)
            cov_matrix = np.cov(swarm.T)

        return {'best_position': global_best_position, 'best_score': global_best_score}