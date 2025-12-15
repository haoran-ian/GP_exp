import numpy as np

class AdvancedQuantumInspiredSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = 60  # Increased population size for diversity
        swarm = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(ind) for ind in swarm])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = np.copy(personal_best_positions[global_best_index])
        global_best_score = personal_best_scores[global_best_index]

        eval_count = population_size

        # Adaptive parameters
        w_max, w_min = 0.8, 0.2  # Adjusted inertia weight range
        c1, c2 = 1.5, 2.5  # Modified cognitive and social coefficients
        CR = 0.8  # Slightly reduced crossover rate

        elite_fraction = 0.15  # Increased elite fraction
        elite_count = max(1, int(elite_fraction * population_size))

        # Covariance Matrix Adaptation parameters
        cov_matrix = np.eye(self.dim)

        while eval_count < self.budget:
            # Linearly reduce inertia weight
            w = w_max - ((w_max - w_min) * (eval_count / self.budget))
            c1_adaptive = c1 * (1 - (eval_count / self.budget)**1.5)  # More aggressive reduction
            c2_adaptive = c2 * ((eval_count / self.budget)**1.5)

            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities + 
                          c1_adaptive * r1 * (personal_best_positions - swarm) + 
                          c2_adaptive * r2 * (global_best_position - swarm))
            swarm += velocities
            swarm = np.clip(swarm, lb, ub)

            # Quantum-Inspired Position Update with increased stochasticity
            sigma = 0.15
            quantum_positions = global_best_position + sigma * np.random.standard_normal((population_size, self.dim))
            quantum_positions = np.clip(quantum_positions, lb, ub)

            # Differential Evolution mutation and crossover with covariance-enhanced recombination
            for i in range(population_size):
                if i >= elite_count:
                    candidates = list(range(population_size))
                    candidates.remove(i)

                    # Combined mutation using elite and covariance influence
                    elite_positions = swarm[:elite_count]
                    elite_mean = np.mean(elite_positions, axis=0)
                    global_influence = global_best_position + np.random.multivariate_normal(np.zeros(self.dim), cov_matrix)
                    mutant = (elite_mean + global_influence + swarm[i]) / 3
                    mutant = np.clip(mutant, lb, ub)

                    crossover = np.random.rand(self.dim) < CR
                    trial = np.where(crossover, mutant, swarm[i])

                    # Quantum-inspired trial update
                    quantum_trial = np.where(np.random.rand(self.dim) < 0.6, trial, quantum_positions[i])

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
            cov_matrix = 0.8 * cov_matrix + 0.2 * np.cov(swarm.T)  # Enhanced adaptation rate

        return {'best_position': global_best_position, 'best_score': global_best_score}