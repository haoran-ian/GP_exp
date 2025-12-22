import numpy as np

class EnhancedHybridSwarmOptimization:
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
        CR = 0.9

        elite_fraction = 0.1
        elite_count = max(1, int(elite_fraction * population_size))

        # Quantum-inspired parameters
        beta = 0.1

        while eval_count < self.budget:
            # Adjust inertia weight and learning coefficients
            w = w_max - ((w_max - w_min) * (eval_count / self.budget))
            c1_adaptive = c1 * (1 - (eval_count / self.budget)**2)
            c2_adaptive = c2 * ((eval_count / self.budget)**2)

            # PSO update
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities + 
                          c1_adaptive * r1 * (personal_best_positions - swarm) + 
                          c2_adaptive * r2 * (global_best_position - swarm))
            swarm += velocities
            swarm = np.clip(swarm, lb, ub)

            # Quantum-inspired mutation
            for i in range(population_size):
                if i < elite_count:
                    continue

                q_bit = np.random.uniform(-beta, beta, self.dim)
                quantum_mutant = swarm[i] + q_bit * (global_best_position - swarm[i])
                quantum_mutant = np.clip(quantum_mutant, lb, ub)

                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, quantum_mutant, swarm[i])

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

            # Dynamic neighborhood search for local refinement
            if eval_count < self.budget:
                for i in range(elite_count):
                    neighbors = np.random.choice(range(population_size), size=5, replace=False)
                    local_best = min(neighbors, key=lambda x: personal_best_scores[x])
                    local_step = np.random.uniform(-0.1, 0.1, self.dim)
                    local_candidate = swarm[local_best] + local_step
                    local_candidate = np.clip(local_candidate, lb, ub)

                    f_local_candidate = func(local_candidate)
                    eval_count += 1

                    if f_local_candidate < personal_best_scores[i]:
                        personal_best_positions[i] = local_candidate
                        personal_best_scores[i] = f_local_candidate
                        if f_local_candidate < global_best_score:
                            global_best_position = local_candidate
                            global_best_score = f_local_candidate

                    if eval_count >= self.budget:
                        break

        return {'best_position': global_best_position, 'best_score': global_best_score}