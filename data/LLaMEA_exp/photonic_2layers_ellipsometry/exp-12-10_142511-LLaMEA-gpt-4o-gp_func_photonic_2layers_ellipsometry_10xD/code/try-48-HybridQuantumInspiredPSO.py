import numpy as np

class HybridQuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        population_size = 10 + 2 * self.dim 
        F_base = 0.5  
        CR_base = 0.9  
        quantum_prob = 0.3  
        
        # Initialize population
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.rand(population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        elite = population[np.argmin(fitness)]

        velocities = np.random.rand(population_size, self.dim) * (ub - lb) * 0.1
        
        while evaluations < self.budget:
            # Calculate population diversity
            population_std = np.std(population, axis=0)
            diversity = np.mean(population_std / (ub - lb))

            # Adaptive parameters
            phase_ratio = evaluations / self.budget
            F = F_base + 0.2 * np.sin(phase_ratio * np.pi)
            CR = CR_base - 0.2 * np.sin(phase_ratio * np.pi)

            swarm_factor = 0.5 + 0.5 * np.cos(2 * np.pi * evaluations / self.budget)

            # Adjust population size and add quantum behavior
            population_size = min(max(5, int(population_size * (1.1 - diversity))), 50)
            quantum_indices = np.random.rand(population_size) < quantum_prob

            for i in range(population_size):
                if quantum_indices[i]:
                    q_mutant = elite + np.random.randn(self.dim) * (diversity + 0.1)
                    q_mutant = np.clip(q_mutant, lb, ub)
                    q_fitness = func(q_mutant)
                    evaluations += 1
                    if q_fitness < fitness[i]:
                        population[i] = q_mutant
                        fitness[i] = q_fitness
                        if q_fitness < func(elite):
                            elite = q_mutant
                    continue

                # Differential mutation with adaptive F
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = swarm_factor * (population[a] + F * (population[b] - population[c]))
                mutant = np.clip(mutant, lb, ub)

                # Crossover with adaptive CR
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < func(elite):
                        elite = trial

            # Update velocities and positions in PSO style
            personal_best = population[np.argmin(fitness)]
            velocities = 0.5 * velocities + 0.5 * np.random.rand(population_size, self.dim) * (personal_best - population)
            population += velocities
            population = np.clip(population, lb, ub)
            
            if evaluations >= self.budget:
                break

        # Return the best found solution
        return elite