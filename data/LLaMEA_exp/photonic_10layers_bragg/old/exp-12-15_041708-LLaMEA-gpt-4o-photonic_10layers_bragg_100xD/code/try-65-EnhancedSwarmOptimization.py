import numpy as np

class EnhancedSwarmOptimization:
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

        # Initial parameters
        w_min, w_max = 0.3, 0.9
        c1, c2 = 2.0, 2.0
        F_base = 0.7
        CR = 0.9

        # Adaptive parameters
        elite_fraction = 0.1
        elite_count = max(1, int(elite_fraction * population_size))

        # Covariance Matrix Adaptation parameters
        cov_matrix = np.eye(self.dim)

        while eval_count < self.budget:
            # Stochastic inertia for enhanced exploration
            w = np.random.uniform(w_min, w_max)
            c1_adaptive = c1 * (1 - (eval_count / self.budget)**2)
            c2_adaptive = c2 * ((eval_count / self.budget)**2)

            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities + 
                          c1_adaptive * r1 * (personal_best_positions - swarm) + 
                          c2_adaptive * r2 * (global_best_position - swarm))
            swarm += velocities
            swarm = np.clip(swarm, lb, ub)

            # Adaptive elite strategy with controlled elite fraction
            elite_fraction = 0.1 + 0.15 * (1 - (global_best_score / max(personal_best_scores)))
            elite_count = max(1, int(elite_fraction * population_size))

            # Hybrid mutation combining DE and PSO strategies
            for i in range(population_size):
                if i >= elite_count:
                    candidates = list(range(population_size))
                    candidates.remove(i)

                    # Hybrid mutation with combined strategies
                    random_candidates = np.random.choice(candidates, 3, replace=False)
                    x1, x2, x3 = swarm[random_candidates]
                    mutant = x1 + F_base * (x2 - x3) + np.random.multivariate_normal(np.zeros(self.dim), cov_matrix)
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

            # Adaptive update of covariance matrix for better local search
            cov_matrix = 0.95 * cov_matrix + 0.05 * np.cov(swarm.T)

        return {'best_position': global_best_position, 'best_score': global_best_score}