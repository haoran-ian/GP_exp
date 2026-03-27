import numpy as np

class AdvancedMultiSwarmHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = min(50, self.budget // 2)
        num_swarms = max(2, population_size // 10)
        swarms = [np.random.uniform(lb, ub, (population_size // num_swarms, self.dim)) for _ in range(num_swarms)]
        fitness = [np.array([func(ind) for ind in swarm]) for swarm in swarms]
        best_indices = [np.argmin(f) for f in fitness]
        best_individuals = [swarm[idx] for swarm, idx in zip(swarms, best_indices)]
        best_fitness = min([f[idx] for f, idx in zip(fitness, best_indices)])
        best_global = best_individuals[np.argmin([f[idx] for f, idx in zip(fitness, best_indices)])]

        evaluations = sum(len(swarm) for swarm in swarms)

        while evaluations < self.budget:
            for s, swarm in enumerate(swarms):
                current_pop_size = max(5, int(len(swarm) * (1 - evaluations / self.budget)))
                F = np.random.uniform(0.5, 0.9)  # Adaptive mutation factor
                CR = np.random.uniform(0.3, 0.8)  # Adaptive crossover rate

                # PSO inertia and acceleration coefficients
                w = np.random.uniform(0.5, 0.9)
                c1 = np.random.uniform(1.5, 2.0)
                c2 = np.random.uniform(1.5, 2.0)
                
                velocities = np.random.uniform(-1, 1, swarm.shape)

                for i in range(current_pop_size):
                    # Update velocity and position using PSO
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities[i] = w * velocities[i] + c1 * r1 * (best_individuals[s] - swarm[i]) + c2 * r2 * (best_global - swarm[i])
                    swarm[i] = np.clip(swarm[i] + velocities[i], lb, ub)
                    
                    # DE mutation and crossover within swarm
                    indices = [idx for idx in range(current_pop_size) if idx != i]
                    a, b, c = swarm[np.random.choice(indices, 3, replace=False)]
                    mutant = np.clip(a + F * (b - c), lb, ub)
                    cross_points = np.random.rand(self.dim) < CR
                    trial = np.where(cross_points, mutant, swarm[i])
                    
                    # Function evaluation
                    trial_fitness = func(trial)
                    evaluations += 1
                    if trial_fitness < fitness[s][i]:
                        swarm[i] = trial
                        fitness[s][i] = trial_fitness
                        
                        if trial_fitness < best_fitness:
                            best_fitness = trial_fitness
                            best_global = trial

                # Enhanced Simulated Annealing-like selection
                T = max(0.01, 1.0 - evaluations / self.budget)
                for i in range(current_pop_size):
                    new_candidate = swarm[i] + np.random.normal(0, 0.05, self.dim)
                    new_candidate = np.clip(new_candidate, lb, ub)
                    new_fitness = func(new_candidate)
                    evaluations += 1
                    if new_fitness < fitness[s][i] or np.random.rand() < np.exp((fitness[s][i] - new_fitness) / T):
                        swarm[i] = new_candidate
                        fitness[s][i] = new_fitness
                        if new_fitness < best_fitness:
                            best_fitness = new_fitness
                            best_global = new_candidate

                if evaluations >= self.budget:
                    break

        return best_global