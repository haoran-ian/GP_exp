import numpy as np

class EnhancedHybridPSODE:
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
        
        # Adaptive inertia weight parameters
        w_max = 0.9
        w_min = 0.4
        c1, c2 = 1.5, 1.5
        F = 0.8  # Differential evolution scaling factor
        CR = 0.9  # Crossover probability

        while eval_count < self.budget:
            # Adaptive inertia weight
            w = w_max - ((w_max - w_min) * (eval_count / self.budget))
            
            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities + 
                          c1 * r1 * (personal_best_positions - swarm) + 
                          c2 * r2 * (global_best_position - swarm))
            swarm += velocities
            swarm = np.clip(swarm, lb, ub)

            # Differential Evolution mutation and crossover
            for i in range(population_size):
                candidates = list(range(population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = np.clip(swarm[a] + F * (swarm[b] - swarm[c]), lb, ub)
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