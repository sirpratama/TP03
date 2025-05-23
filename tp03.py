from collections import deque
import heapq

class Planet: # Initialize Planet class
    def __init__(self, planet_id, is_prime):
        self.id = planet_id
        self.is_prime = is_prime
        self.connections = []

    def add_connection(self, neighbor_id, energy_cost):
        self.connections.append((neighbor_id, energy_cost))

class GalacticGraph:
    def __init__(self, n, planet_types_str):
        # Initialize planets list
        self.planets = []

        # Split the planet_types_str string into a list of types
        parsed_planet_types = planet_types_str.split()

        # Create planets
        for i in range(n):
            # Ensure we don't go out of bounds if n is larger than number of types provided
            if i < len(parsed_planet_types):
                is_prime = (parsed_planet_types[i] == "P")
            else:
                # Default to not prime if not enough type specifiers, or handle error
                is_prime = False 
            self.planets.append(Planet(i, is_prime)) # Planets are 0-indexed internally
        self.all_edges_for_kruskal = []

    # Add warp lane (Edges)
    def add_warp_lane(self, u, v, energy):
        # Adjust to 0-indexed before accessing self.planets
        self.planets[u-1].add_connection(v, energy) # v is 1-indexed ID for connection
        self.planets[v-1].add_connection(u, energy) # u is 1-indexed ID for connection
        # Add edge to the Kruskal edge list with 0-indexed planet IDs
        self.all_edges_for_kruskal.append((energy, u-1, v-1))

    # Get prime planets
    def get_prime_planets(self):
        # Returns 1-indexed IDs
        return [i+1 for i, planet in enumerate(self.planets) if planet.is_prime]

# For Kruskal's Algorithm
class DisjointSet:
    def __init__(self, n):
        # Parent array, initially each planet is its own parent
        self.parent = list(range(n))
        # Rank array for union by rank optimization
        self.rank = [0] * n
    
    def find(self, x):
        # Find the root of the set containing x
        # Path compression: make every examined node point directly to the root
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        # Merge the sets containing x and y
        # Union by rank: attach smaller tree under root of larger tree
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:  # Only proceed if they are in different sets
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                # Make root_x the parent of root_y (or vice versa)
                self.parent[root_y] = root_x
                # Increment rank of root_x
                self.rank[root_x] += 1
            return True  # Indicate that a union was performed
        return False  # Indicate that they were already in the same set

# Mission A - Initial Exploration
# Count reachable Prime Planets with BFS
def mission_a(graph, start_planet_id, initial_energy): # start_planet_id is 1-indexed
    planets = graph.planets  # Cache reference to avoid repeated attribute lookup
    n = len(planets)
    visited = [False] * n
    queue = deque([start_planet_id - 1]) 
    visited[start_planet_id - 1] = True
    prime_count = 0

    while queue:
        current_idx = queue.popleft() # 0-indexed
        current_planet = planets[current_idx]

        if current_planet.is_prime:
            prime_count += 1
        
        for neighbor_id, energy_required in current_planet.connections: # neighbor_id is 1-indexed
            if energy_required <= initial_energy:  # Check energy constraint first
                neighbor_idx = neighbor_id - 1  # Calculate once and reuse
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    queue.append(neighbor_idx) # Add 0-indexed to queue

    return prime_count

# Mission B - Rescue Operation
# Find minimum energy needed to reach any Prime Planet from start_planet_id
def mission_b(graph, start_planet_id): # start_planet_id is 1-indexed
    n = len(graph.planets)
    min_energy = [float('inf')] * n
    min_energy[start_planet_id-1] = 0 # Use 0-indexed for min_energy array

    # Priority queue: (current_max_energy, planet_idx_0_indexed)
    pq = [(0, start_planet_id-1)] 
    visited = [False] * n

    while pq:
        energy, current_idx = heapq.heappop(pq) # current_idx is 0-indexed

        if visited[current_idx]:
            continue
        visited[current_idx] = True

        if graph.planets[current_idx].is_prime:
            return energy
        
        for neighbor_id, edge_energy in graph.planets[current_idx].connections: # neighbor_id is 1-indexed
            new_max_energy = max(energy, edge_energy)
            # Use 0-indexed for min_energy and visited arrays
            if new_max_energy < min_energy[neighbor_id-1]:
                min_energy[neighbor_id-1] = new_max_energy
                heapq.heappush(pq, (new_max_energy, neighbor_id-1)) # Push 0-indexed to pq

    return -1

def mission_c(graph):
    # Edges are already prepared in graph.all_edges_for_kruskal
    # Stored as (energy, u_idx_0_indexed, v_idx_0_indexed)
    edges = list(graph.all_edges_for_kruskal) # Make a copy to sort
            
    edges.sort()
    n = len(graph.planets)
    disjoint_set = DisjointSet(n) # Works with 0 to n-1
    total_energy = 0
    mst_edges = 0

    for energy, u_idx, v_idx in edges: # u_idx, v_idx are 0-indexed
        if disjoint_set.union(u_idx, v_idx): # union now returns True if a merge happened
            total_energy += energy
            mst_edges += 1
            if mst_edges == n - 1:
                break
    return total_energy

def mission_d(graph, start_planet_id, end_planet_id): # IDs are 1-indexed
    n = len(graph.planets)
    prime_planet_ids = graph.get_prime_planets() # Returns 1-indexed IDs

    # dijkstra_standard expects 1-indexed start_planet_id
    energy_from_start = dijkstra_standard(graph, start_planet_id) # Returns list indexed 0 to n-1
    energy_to_end = dijkstra_standard(graph, end_planet_id) # Returns list indexed 0 to n-1
    
    # Access energy arrays with 0-indexed (id-1)
    min_energy = energy_from_start[end_planet_id-1]

    for prime_id in prime_planet_ids: # prime_id is 1-indexed
        # Access energy arrays with 0-indexed (id-1)
        if energy_from_start[prime_id-1] != float('inf') and energy_to_end[prime_id-1] != float('inf'):
            total_energy = energy_from_start[prime_id-1] + energy_to_end[prime_id-1]
            min_energy = min(min_energy, total_energy)
    return min_energy

def dijkstra_standard(graph, start_planet_id): # start_planet_id is 1-indexed
    n = len(graph.planets)
    min_energy = [float('inf')] * n
    min_energy[start_planet_id-1] = 0 # Use 0-indexed for array access

    # pq stores (energy, planet_idx_0_indexed)
    pq = [(0, start_planet_id-1)] 
    visited = [False] * n

    while pq:
        energy, current_idx = heapq.heappop(pq) # current_idx is 0-indexed

        if visited[current_idx]:
            continue
        visited[current_idx] = True

        for neighbor_id, edge_energy in graph.planets[current_idx].connections: # neighbor_id is 1-indexed
            new_energy = energy + edge_energy
            # Use 0-indexed for array access (neighbor_id-1)
            if new_energy < min_energy[neighbor_id-1]:
                min_energy[neighbor_id-1] = new_energy
                heapq.heappush(pq, (new_energy, neighbor_id-1)) # Push 0-indexed to pq
    return min_energy # Returns list indexed 0 to n-1
    

def main():
    # Read input
    n, m = map(int, input().split())
    planet_types_str = input() # Keep as string
    
    # Create graph
    graph = GalacticGraph(n, planet_types_str)
    
    # Add warp lanes
    for _ in range(m):
        u, v, e = map(int, input().split()) # u,v are 1-indexed
        graph.add_warp_lane(u, v, e)
    
    # Process missions
    q = int(input())
    for _ in range(q):
        command = input().split()
        
        if command[0] == 'A':
            energy = int(command[1])
            # mission_a expects 1-indexed start planet. Example implies start is Planet 1.
            result = mission_a(graph, 1, energy) 
            print(result)
        
        elif command[0] == 'B':
            s_id = int(command[1]) # 1-indexed
            t_id = int(command[2]) # 1-indexed
            # Access graph.planets with 0-indexed t_id-1
            if graph.planets[t_id-1].is_prime:
                print(0)
            else:
                # mission_b expects 1-indexed start planet
                result = mission_b(graph, s_id) 
                print(result)
        
        elif command[0] == 'C':
            result = mission_c(graph)
            print(result)
        
        elif command[0] == 'D':
            g_id = int(command[1]) # 1-indexed
            d_id = int(command[2]) # 1-indexed
            # mission_d expects 1-indexed start/end planets
            result = mission_d(graph, g_id, d_id) 
            print(result)

if __name__ == "__main__":
    main()