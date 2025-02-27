import random
import uuid

class Individual:
    def __init__(self, genome: str,
                 parent = None,
                 generation: int = 0,
                 mutation_rate: float = 0.01):
        self.id = str(uuid.uuid4())
        self.genome = genome
        self.parent = parent  # Direct parent
        self.ancestors = []  # List of ancestor IDs
        self.generation = generation
        self.mutation_rate = mutation_rate
        
        if parent is not None:
            self.ancestors = parent.ancestors.copy()
            self.ancestors.append(parent.id)
    
    def mutate(self) -> 'Individual':
        """Create a descendant with mutation"""
        new_genome = list(self.genome)  # convert to list<char>
        for i in range(len(new_genome)):
            if random.random() < self.mutation_rate:
                new_genome[i] = random.choice('ACTG')
        
        # Construct and return new mutated individual
        return Individual(
            genome=''.join(new_genome),
            parent=self,
            generation=self.generation + 1,
            mutation_rate=self.mutation_rate
            )
    
    def get_genetic_distance(self, other: 'Individual') -> int:
        """Calculate genetic distance (number of different characters)"""
        # Genetic distance as just the number of different nucleaotides
        return sum(c1 != c2 for c1, c2 in zip(self.genome, other.genome))
    
    def __str__(self):
        """Return first 8 digits of uuid as representation of individual"""
        return f"Individual {self.id[:8]} (Gen {self.generation})"

