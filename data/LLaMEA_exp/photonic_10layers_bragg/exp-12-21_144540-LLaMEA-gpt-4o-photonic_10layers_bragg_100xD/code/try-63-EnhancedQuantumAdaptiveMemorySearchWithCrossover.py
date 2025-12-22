import numpy as np

class EnhancedQuantumAdaptiveMemorySearchWithCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)  # Fixed memory size
        self.memory = []  # Store solutions
        self.iteration = 0
        self.adaptivity_rate = 0.1  # Initial adaptivity rate for exploration/exploitation balance
        self.phase_shift_rate = 0.1  # Phase shift diversification parameter
        self.crossover_rate = 0.5  # Probability for crossover operation

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        for _ in range(self.memory_size):
            solution = np.random.uniform(lb, ub, self.dim)
            value = func(solution)
            self._update_memory(solution, value)
            if value < best_value:
                best_value = value
                best_solution = solution
        
        self.iteration += self.memory_size

        while self.iteration < self.budget:
            memory_solution, memory_value = self._select_from_memory()
            self.adaptivity_rate = min(0.5, self.adaptivity_rate + 0.01 * (best_value - memory_value))
            
            if np.random.rand() < self.crossover_rate:
                # Apply crossover strategy
                crossover_solution = self._crossover(memory_solution)
                candidate_solution = self._quantum_explore(lb, ub, crossover_solution)
            else:
                candidate_solution = self._quantum_explore(lb, ub)

            candidate_value = func(candidate_solution)

            # Apply phase shift for diversification
            phase_shift_candidate = self._phase_shift(candidate_solution, lb, ub)
            phase_shift_value = func(phase_shift_candidate)
            candidate_solution, candidate_value = (phase_shift_candidate, phase_shift_value) if phase_shift_value < candidate_value else (candidate_solution, candidate_value)

            self.iteration += 1

            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            self._update_memory(candidate_solution, candidate_value)

        return best_solution

    def _select_from_memory(self):
        probabilities = np.array(
            [self.adaptivity_rate if i == 0 else np.random.rand() for i in range(len(self.memory))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _quantum_explore(self, lb, ub, crossover_solution=None):
        superposition = np.array([np.mean([sol for sol, _ in self.memory], axis=0)])
        entanglement = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim) * (ub - lb)
        candidate_solution = superposition + entanglement
        if crossover_solution is not None:
            candidate_solution = (candidate_solution + crossover_solution) / 2
        return np.clip(candidate_solution, lb, ub).flatten()

    def _phase_shift(self, solution, lb, ub):
        shift = np.random.uniform(-self.phase_shift_rate, self.phase_shift_rate, self.dim) * (ub - lb)
        shifted_solution = solution + shift
        return np.clip(shifted_solution, lb, ub)

    def _crossover(self, solution):
        partner_idx = np.random.randint(0, len(self.memory))
        partner_solution = self.memory[partner_idx][0]
        mask = np.random.rand(self.dim) < self.crossover_rate
        new_solution = np.where(mask, solution, partner_solution)
        return new_solution

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])
        if len(self.memory) > self.memory_size:
            self.memory.pop()