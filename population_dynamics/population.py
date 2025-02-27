import random
from typing import List, Dict, Tuple, Optional
from individual import Individual

class Population:
    def __init__(self,
                 initial_size: int = 10,
                 genome_length: int = 500,
                 mutation_rate: float = 0.01):

        self.individuals: List[Individual] = []
        self.generations: List[List[Individual]] = []
        self.genome_length = genome_length
        self.mutation_rate = mutation_rate
        
        # Generate initial population
        self.initialize_population(initial_size)
    
    def initialize_population(self, population_size: int):
        """Generate initial population with random genomes"""
        initial_generation = []
        for _ in range(population_size):
            genome = ''.join(random.choice('ACTG') for _ in range(self.genome_length))
            individual = Individual(genome=genome, mutation_rate=self.mutation_rate)
            initial_generation.append(individual)
        
        self.individuals = initial_generation
        self.generations = [initial_generation]
    
    def get_next_generation(self, new_size: Optional[int] = None):
        """Generate the next generation through reproduction and mutation"""
        if new_size is None:
            new_size = len(self.individuals)
        
        # Select parents (with replacement) and create offspring
        parents = random.choices(self.individuals, k=new_size)  # with replacement
        # parents = random.sample(self.individuals, k=new_size) # without replacement, k cannot be larger than len(self.individuals)
        new_generation = [parent.mutate() for parent in parents]
        
        self.individuals = new_generation
        self.generations.append(new_generation)
        
        return new_generation
    
    def find_most_recent_common_ancestor(self, individuals: List[Individual]) -> Tuple[Optional[str], int]:
        """Find the most recent common ancestor of given individuals"""
        if not individuals or len(individuals) < 2:
            return None, -1
        
        # Get sets of ancestors for each individual
        ancestor_sets = []
        for individual in individuals:
            ancestors = set(individual.ancestors)
            ancestors.add(individual.id)  # Include the individual itself
            ancestor_sets.append(ancestors)
        
        # Find common ancestors
        common_ancestors = set.intersection(*ancestor_sets)
        
        if not common_ancestors:
            return None, -1
        
        # Find the most recent common ancestor
        most_recent_gen = -1
        most_recent_ancestor_id = None
        
        for gen_idx, generation in enumerate(self.generations):
            for individual in generation:
                if individual.id in common_ancestors and gen_idx > most_recent_gen:
                    most_recent_gen = gen_idx
                    most_recent_ancestor_id = individual.id
        
        return most_recent_ancestor_id, most_recent_gen
    
    def get_individual_by_id(self, individual_id: str) -> Optional[Individual]:
        """Get individual by ID from all generations"""
        for generation in self.generations:
            for individual in generation:
                if individual.id == individual_id:
                    return individual
        return None
    
    def time_to_most_recent_common_ancestor(self) -> int:
        """Calculate how many generations ago all current individuals had a common ancestor"""
        mrca_id, mrca_gen = self.find_most_recent_common_ancestor(self.individuals)
        if mrca_id is None:
            return -1
        
        current_gen = len(self.generations) - 1
        return current_gen - mrca_gen
    
    def get_ancestor_trace(self, individual: Individual) -> List[Individual]:
        """Get the complete ancestry trace of an individual"""
        trace = []
        for ancestor_id in individual.ancestors:
            ancestor = self.get_individual_by_id(ancestor_id)
            if ancestor:
                trace.append(ancestor)
        return trace
    
    def analyze_population_size_vs_mrca_time(self, sizes: List[int], repetitions: int = 5) -> Dict:
        """Analyze relationship between population size and time to MRCA"""
        results = {size: [] for size in sizes}
        
        current_state = {
            'individuals': self.individuals.copy(),
            'generations': [g.copy() for g in self.generations]
        }
        
        for size in sizes:
            for _ in range(repetitions):
                # Reset population to a single ancestor
                self.initialize_population(size)
                
                # Evolve for a significant number of generations
                for _ in range(max(50, size * 2)):
                    self.get_next_generation(size)
                
                # Calculate TMRCA
                tmrca = self.time_to_most_recent_common_ancestor()
                if tmrca > 0:  # Only include valid results
                    results[size].append(tmrca)
        
        # Restore original state
        self.individuals = current_state['individuals']
        self.generations = current_state['generations']
        
        # Calculate averages
        averages = {size: sum(times)/len(times) if times else 0 for size, times in results.items()}
        
        return averages
