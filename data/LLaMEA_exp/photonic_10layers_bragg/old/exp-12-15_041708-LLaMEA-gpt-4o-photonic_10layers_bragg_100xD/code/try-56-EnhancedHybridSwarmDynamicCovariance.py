import numpy as np

class EnhancedHybridSwarmDynamicCovariance:
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
        CR_base = 0.9

        elite_fraction = 0.1
        elite_count = max(1, int(elite_fraction * population_size))

        # Covariance Matrix Adaptation parameters
        cov_matrix = np.eye(self.dim)

        while eval_count < self.budget:
            # Dynamically adjust inertia weight
            w = w_max - ((w_max - w_min) * (eval_count / self.budget))

            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities + 
                          c1 * r1 * (personal_best_positions - swarm) + 
                          c2 * r2 * (global_best_position - swarm))
            swarm += velocities
            swarm = np.clip(swarm, lb, ub)

            # Dynamic elite niching based on global best improvement
            if global_best_score < min(personal_best_scores):
                elite_fraction += 0.05
            else:
                elite_fraction -= 0.05
            elite_fraction = np.clip(elite_fraction, 0.1, 0.3)
            elite_count = max(1, int(elite_fraction * population_size))

            # Adaptive crossover rate based on fitness landscape
            CR = CR_base * (1 - (global_best_score / max(personal_best_scores)))

            # Differential Evolution mutation and crossover with dynamic covariance
            for i in range(population_size):
                candidates = list(range(population_size))
                candidates.remove(i)

                if i >= elite_count:
                    # Mutation
                    elite_positions = swarm[:elite_count]
                    elite_mean = np.mean(elite_positions, axis=0)
                    global_influence = global_best_position + np.random.multivariate_normal(np.zeros(self.dim), cov_matrix)
                    mutant = (elite_mean + global_influence + swarm[i]) / 3
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

            # Update covariance matrix for better local search
            cov_matrix = 0.85 * cov_matrix + 0.15 * np.cov(swarm.T)

        return {'best_position': global_best_position, 'best_score': global_best_score}