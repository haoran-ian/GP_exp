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
        F_base = 0.8  # Base DE scaling factor
        CR = 0.9  # Crossover probability
        local_search_rate = 0.1 * self.dim  # Controls local search intensity

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

            # Differential Evolution mutation and crossover with adaptive strategy
            for i in range(population_size):
                candidates = list(range(population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                
                # Self-adaptive mutation strategy
                F = F_base + 0.2 * np.random.rand() * ((self.budget - eval_count) / self.budget)  # Decaying factor added
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

                # Dynamic local search to refine personal bests
                if np.random.rand() < local_search_rate / self.dim:
                    local_search_candidate = personal_best_positions[i] + 0.1 * np.random.normal(size=self.dim)
                    local_search_candidate = np.clip(local_search_candidate, lb, ub)
                    f_local = func(local_search_candidate)
                    eval_count += 1
                    if f_local < personal_best_scores[i]:
                        personal_best_positions[i] = local_search_candidate
                        personal_best_scores[i] = f_local
                        if f_local < global_best_score:
                            global_best_position = local_search_candidate
                            global_best_score = f_local

                if eval_count >= self.budget:
                    break

        return {'best_position': global_best_position, 'best_score': global_best_score}