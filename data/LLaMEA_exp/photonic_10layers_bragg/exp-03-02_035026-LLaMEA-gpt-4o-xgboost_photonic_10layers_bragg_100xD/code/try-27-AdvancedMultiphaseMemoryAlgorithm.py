import numpy as np

class AdvancedMultiphaseMemoryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = 5  # Number of solutions to maintain in memory
        self.memory = []  # Stores best solutions found

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, size=self.dim)
        best_fitness = func(best_solution)
        evaluations = 1

        while evaluations < self.budget:
            phase = evaluations / self.budget

            # Adaptive scale based on phase and solution memory
            scale = self._adaptive_scale(phase, best_fitness, len(self.memory))

            candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale)
            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)

            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]
            
            # Update memory with the best solution of this round
            self._update_memory(best_solution, best_fitness)
        
        return best_solution
    
    def _generate_solutions(self, center, lb, ub, scale):
        # Generates new candidate solutions around a center point
        perturbations = np.random.uniform(-scale, scale, size=(10, self.dim))
        solutions = center + perturbations * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _adaptive_scale(self, phase, best_fitness, memory_size):
        # Adaptive perturbation variance based on the phase and memory diversity
        base_scale = 0.2 * self._fitness_variance(best_fitness) / np.sqrt(self.dim)
        if phase < 0.3:  # Exploration Phase
            return base_scale * (1 + 0.5 * memory_size / self.memory_size)
        elif phase < 0.7:  # Balanced Phase
            return base_scale * (1 + 0.2 * memory_size / self.memory_size)
        else:  # Exploitation Phase
            return base_scale * (1 - 0.3 * memory_size / self.memory_size)

    def _fitness_variance(self, best_fitness):
        return max(0.1, np.abs(best_fitness) / 10)

    def _update_memory(self, solution, fitness):
        # Update memory, keeping only the top solutions
        self.memory.append((solution, fitness))
        self.memory.sort(key=lambda x: x[1])  # Sort by fitness
        if len(self.memory) > self.memory_size:
            self.memory.pop()  # Remove the worst if memory is full