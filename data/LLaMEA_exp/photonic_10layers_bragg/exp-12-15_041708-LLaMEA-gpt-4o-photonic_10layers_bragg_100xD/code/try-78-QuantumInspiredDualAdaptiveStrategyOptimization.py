import numpy as np

class QuantumInspiredDualAdaptiveStrategyOptimization:
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

        # Covariance Matrix Adaptation parameters
        cov_matrix = np.eye(self.dim)

        while eval_count < self.budget:
            # Linearly reduce inertia weight
            w = w_max - ((w_max - w_min) * (eval_count / self.budget))
            c1_adaptive = c1 * (1 - (eval_count / self.budget)**2)
            c2_adaptive = c2 * ((eval_count / self.budget)**2)

            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities + 
                          c1_adaptive * r1 * (personal_best_positions - swarm) + 
                          c2_adaptive * r2 * (global_best_position - swarm))
            swarm += velocities
            swarm = np.clip(swarm, lb, ub)

            # Quantum-Inspired Position Update with dual adaptive strategy
            sigma_initial = 0.1
            sigma_decay = 0.99
            sigma = sigma_initial * (sigma_decay ** (eval_count / self.budget))
            quantum_positions = global_best_position + sigma * np.random.standard_normal((population_size, self.dim))
            quantum_positions = np.clip(quantum_positions, lb, ub)

            # Differential Evolution mutation and crossover with covariance-enhanced recombination
            for i in range(population_size):
                candidates = list(range(population_size))
                candidates.remove(i)

                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = swarm[a] + 0.8 * (swarm[b] - swarm[c])
                mutant = np.clip(mutant, lb, ub)

                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, swarm[i])

                # Enhanced quantum-inspired trial update
                quantum_trial = np.where(np.random.rand(self.dim) < 0.5, trial, quantum_positions[i])

                f_trial = func(quantum_trial)
                eval_count += 1

                if f_trial < personal_best_scores[i]:
                    personal_best_positions[i] = quantum_trial
                    personal_best_scores[i] = f_trial
                    if f_trial < global_best_score:
                        global_best_position = quantum_trial
                        global_best_score = f_trial

                if eval_count >= self.budget:
                    break

            # Update covariance matrix adaptively for better local search
            cov_matrix = 0.9 * cov_matrix + 0.1 * np.cov(swarm.T)

        return {'best_position': global_best_position, 'best_score': global_best_score}