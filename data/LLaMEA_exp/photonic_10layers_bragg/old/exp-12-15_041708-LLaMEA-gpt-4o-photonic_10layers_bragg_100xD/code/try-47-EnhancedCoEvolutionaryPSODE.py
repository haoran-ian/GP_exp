import numpy as np

class EnhancedCoEvolutionaryPSODE:
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
        
        elite_fraction = 0.1  # Retain the top 10% best solutions
        elite_count = max(1, int(elite_fraction * population_size))

        cov_matrix = np.eye(self.dim)
        mean = np.mean(swarm, axis=0)
        
        while eval_count < self.budget:
            # Dynamic inertia weight and adaptive learning factors
            w = w_max - ((w_max - w_min) * (eval_count / self.budget))
            F_adaptive = F_base * (1 + np.sin(2 * np.pi * eval_count / self.budget))
            c1_adaptive, c2_adaptive = c1 * (1 + np.cos(2 * np.pi * eval_count / self.budget)), c2

            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities + 
                          c1_adaptive * r1 * (personal_best_positions - swarm) + 
                          c2_adaptive * r2 * (global_best_position - swarm))
            swarm += velocities
            swarm = np.clip(swarm, lb, ub)

            # Updated elite fraction adjustment
            elite_fraction = max(0.1, 0.1 + 0.3 * (eval_count / self.budget))
            elite_count = max(1, int(elite_fraction * population_size))

            # Differential Evolution mutation and crossover influenced by elite covariance
            for i in range(population_size):
                if i >= elite_count:
                    candidates = list(range(population_size))
                    candidates.remove(i)

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

            # Update mean and covariance matrix with elite information
            mean = np.mean(swarm, axis=0)
            cov_matrix = np.cov(swarm.T) * (1 + np.var([personal_best_scores[i] for i in range(elite_count)]))

        return {'best_position': global_best_position, 'best_score': global_best_score}