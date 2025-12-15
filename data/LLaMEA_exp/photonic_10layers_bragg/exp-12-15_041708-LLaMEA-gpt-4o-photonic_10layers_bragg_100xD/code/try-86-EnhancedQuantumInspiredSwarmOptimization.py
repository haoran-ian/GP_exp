import numpy as np

class EnhancedQuantumInspiredSwarmOptimization:
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
        c1_init, c2_init = 2.5, 1.5
        CR = 0.9  # Crossover rate

        elite_fraction = 0.1
        elite_count = max(1, int(elite_fraction * population_size))

        # Learning rate adaptation
        learning_rate_decay = 0.99

        while eval_count < self.budget:
            # Reduce inertia weight
            w = w_max - ((w_max - w_min) * (eval_count / self.budget))
            c1 = c1_init * (learning_rate_decay ** (eval_count / self.budget))
            c2 = c2_init * ((1 - learning_rate_decay) ** (eval_count / self.budget))

            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities +
                          c1 * r1 * (personal_best_positions - swarm) +
                          c2 * r2 * (global_best_position - swarm))
            swarm += velocities
            swarm = np.clip(swarm, lb, ub)

            # Quantum-Inspired Position Update
            sigma = 0.1
            quantum_positions = global_best_position + sigma * np.random.standard_normal((population_size, self.dim))
            quantum_positions = np.clip(quantum_positions, lb, ub)

            # Differential Evolution mutation and crossover
            for i in range(population_size):
                if i >= elite_count:
                    candidates = list(range(population_size))
                    candidates.remove(i)

                    # Mutant creation using elite influence
                    elite_positions = swarm[:elite_count]
                    elite_mean = np.mean(elite_positions, axis=0)
                    mutant = (elite_mean + swarm[i]) / 2
                    mutant = np.clip(mutant, lb, ub)

                    crossover = np.random.rand(self.dim) < CR
                    trial = np.where(crossover, mutant, swarm[i])

                    # Quantum-inspired trial update
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

        return {'best_position': global_best_position, 'best_score': global_best_score}