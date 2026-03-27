import numpy as np

class AdvancedHybridGA_PSO_SADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.cr = 0.9
        self.f = 0.5
        self.cr_memory = [0.1, 0.2, 0.5, 0.9]
        self.f_memory = [0.4, 0.6, 0.8, 1.0]
        self.mutation_rate = 0.1
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        vel = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = np.copy(pop)
        personal_best_value = np.array([func(ind) for ind in pop])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size

        while evaluations < self.budget:
            # Update velocities and positions (PSO)
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            vel = self.w * vel + self.c1 * r1 * (personal_best - pop) + self.c2 * r2 * (global_best - pop)
            pop = pop + vel
            pop = np.clip(pop, lb, ub)

            # Evaluate new positions
            new_values = np.array([func(ind) for ind in pop])
            evaluations += self.population_size

            # Update personal and global bests
            improvement = new_values < personal_best_value
            personal_best[improvement] = pop[improvement]
            personal_best_value[improvement] = new_values[improvement]

            if np.min(personal_best_value) < global_best_value:
                global_best = personal_best[np.argmin(personal_best_value)]
                global_best_value = np.min(personal_best_value)

            # Genetic Algorithm - Crossover and Mutation
            for i in range(0, self.population_size, 2):
                if evaluations >= self.budget:
                    break
                parent1, parent2 = pop[i], pop[(i+1) % self.population_size]
                crossover_point = np.random.randint(1, self.dim)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                child1 = np.clip(child1, lb, ub)
                child2 = np.clip(child2, lb, ub)

                if np.random.rand() < self.mutation_rate:
                    mutation_idx = np.random.randint(0, self.dim)
                    child1[mutation_idx] = np.random.uniform(lb[mutation_idx], ub[mutation_idx])
                
                if np.random.rand() < self.mutation_rate:
                    mutation_idx = np.random.randint(0, self.dim)
                    child2[mutation_idx] = np.random.uniform(lb[mutation_idx], ub[mutation_idx])
                
                child1_value = func(child1)
                child2_value = func(child2)
                evaluations += 2

                if child1_value < personal_best_value[i]:
                    pop[i] = child1
                    personal_best[i] = child1
                    personal_best_value[i] = child1_value
                    if child1_value < global_best_value:
                        global_best = child1
                        global_best_value = child1_value

                if child2_value < personal_best_value[(i+1) % self.population_size]:
                    pop[(i+1) % self.population_size] = child2
                    personal_best[(i+1) % self.population_size] = child2
                    personal_best_value[(i+1) % self.population_size] = child2_value
                    if child2_value < global_best_value:
                        global_best = child2
                        global_best_value = child2_value

            # Self-Adaptive Differential Evolution (SADE)
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                memory_idx = np.random.choice(len(self.cr_memory))
                self.cr = self.cr_memory[memory_idx]
                self.f = self.f_memory[memory_idx]

                mutant = np.clip(a + self.f * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, pop[i])
                trial_value = func(trial)
                evaluations += 1

                if trial_value < new_values[i]:
                    pop[i] = trial
                    new_values[i] = trial_value
                    self.cr_memory[memory_idx] = (self.cr_memory[memory_idx] + self.cr) / 2
                    self.f_memory[memory_idx] = (self.f_memory[memory_idx] + self.f) / 2
                    if trial_value < personal_best_value[i]:
                        personal_best[i] = trial
                        personal_best_value[i] = trial_value
                        if trial_value < global_best_value:
                            global_best = trial
                            global_best_value = trial_value

        return global_best