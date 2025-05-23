from collections import deque
import heapq

class Planet:
    def __init__(self, planet_id, is_prime):
        self.id = planet_id
        self.is_prime = is_prime
        self.connections = []

    def add_connection(self, neighbor_id, energy_cost):
        self.connections.append((neighbor_id, energy_cost))

class GalacticGraph:
    def __init__(self, n, planet_types_str):
        self.planets = []
        parsed_planet_types = planet_types_str.split()
        
        for i in range(n):
            if i < len(parsed_planet_types):
                is_prime = (parsed_planet_types[i] == "P")
            else:
                is_prime = False 
            self.planets.append(Planet(i, is_prime))
        self.all_edges_for_kruskal = []

    def add_warp_lane(self, u, v, energy):
        self.planets[u-1].add_connection(v, energy)
        self.planets[v-1].add_connection(u, energy)
        self.all_edges_for_kruskal.append((energy, u-1, v-1))

    def get_prime_planets(self):
        return [i+1 for i, planet in enumerate(self.planets) if planet.is_prime]

class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
            return True
        return False

# Mission A - Optimized with early prime counting
def mission_a(graph, start_planet_id, initial_energy):
    planets = graph.planets
    n = len(planets)
    visited = [False] * n
    queue = deque([start_planet_id - 1])
    visited[start_planet_id - 1] = True
    prime_count = 0

    while queue:
        current_idx = queue.popleft()
        current_planet = planets[current_idx]

        if current_planet.is_prime:
            prime_count += 1
        
        for neighbor_id, energy_required in current_planet.connections:
            if energy_required <= initial_energy:
                neighbor_idx = neighbor_id - 1
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    queue.append(neighbor_idx)

    return prime_count

# Mission B - Optimized with early termination
def mission_b(graph, start_planet_id):
    n = len(graph.planets)
    min_energy = [float('inf')] * n
    min_energy[start_planet_id-1] = 0
    
    pq = [(0, start_planet_id-1)]
    visited = [False] * n

    while pq:
        energy, current_idx = heapq.heappop(pq)

        if visited[current_idx]:
            continue
        visited[current_idx] = True

        # Early termination when we reach a prime planet
        if graph.planets[current_idx].is_prime:
            return energy
        
        for neighbor_id, edge_energy in graph.planets[current_idx].connections:
            new_max_energy = max(energy, edge_energy)
            neighbor_idx = neighbor_id - 1
            if not visited[neighbor_idx] and new_max_energy < min_energy[neighbor_idx]:
                min_energy[neighbor_idx] = new_max_energy
                heapq.heappush(pq, (new_max_energy, neighbor_idx))

    return -1

def mission_c(graph):
    edges = list(graph.all_edges_for_kruskal)
    edges.sort()
    n = len(graph.planets)
    disjoint_set = DisjointSet(n)
    total_energy = 0
    mst_edges = 0

    for energy, u_idx, v_idx in edges:
        if disjoint_set.union(u_idx, v_idx):
            total_energy += energy
            mst_edges += 1
            if mst_edges == n - 1:
                break
    return total_energy

# Mission D - Optimized to O(P) instead of O(P²)
def mission_d(graph, start_planet_id, end_planet_id):
    n = len(graph.planets)
    prime_planet_ids = graph.get_prime_planets()
    
    # Run Dijkstra from start and end
    energy_from_start = dijkstra_standard(graph, start_planet_id)
    energy_to_end = dijkstra_standard(graph, end_planet_id)
    
    # Option 1: Direct path
    min_energy = energy_from_start[end_planet_id-1]
    
    # Option 2: Use quantum leap - optimized approach
    # Find minimum cost to reach ANY prime from start
    min_to_any_prime = float('inf')
    for prime_id in prime_planet_ids:
        cost = energy_from_start[prime_id-1]
        if cost < min_to_any_prime:
            min_to_any_prime = cost
    
    # Find minimum cost from ANY prime to end
    min_from_any_prime = float('inf')
    for prime_id in prime_planet_ids:
        cost = energy_to_end[prime_id-1]
        if cost < min_from_any_prime:
            min_from_any_prime = cost
    
    # Total cost using quantum leap
    if min_to_any_prime != float('inf') and min_from_any_prime != float('inf'):
        quantum_cost = min_to_any_prime + min_from_any_prime
        min_energy = min(min_energy, quantum_cost)
    
    return min_energy

# Optimized Dijkstra with better performance
def dijkstra_standard(graph, start_planet_id):
    n = len(graph.planets)
    min_energy = [float('inf')] * n
    min_energy[start_planet_id-1] = 0
    
    pq = [(0, start_planet_id-1)]
    visited = [False] * n
    visited_count = 0

    while pq and visited_count < n:
        energy, current_idx = heapq.heappop(pq)

        if visited[current_idx]:
            continue
        visited[current_idx] = True
        visited_count += 1

        # Skip if this path is already worse than what we found
        if energy > min_energy[current_idx]:
            continue

        for neighbor_id, edge_energy in graph.planets[current_idx].connections:
            new_energy = energy + edge_energy
            neighbor_idx = neighbor_id - 1
            if not visited[neighbor_idx] and new_energy < min_energy[neighbor_idx]:
                min_energy[neighbor_idx] = new_energy
                heapq.heappush(pq, (new_energy, neighbor_idx))
    
    return min_energy

def main():
    n, m = map(int, input().split())
    planet_types_str = input()
    
    graph = GalacticGraph(n, planet_types_str)
    
    for _ in range(m):
        u, v, e = map(int, input().split())
        graph.add_warp_lane(u, v, e)
    
    q = int(input())
    for _ in range(q):
        command = input().split()
        
        if command[0] == 'A':
            energy = int(command[1])
            result = mission_a(graph, 1, energy)
            print(result)
        
        elif command[0] == 'B':
            s_id = int(command[1])
            t_id = int(command[2])
            if graph.planets[t_id-1].is_prime:
                print(0)
            else:
                result = mission_b(graph, s_id)
                print(result)
        
        elif command[0] == 'C':
            result = mission_c(graph)
            print(result)
        
        elif command[0] == 'D':
            g_id = int(command[1])
            d_id = int(command[2])
            result = mission_d(graph, g_id, d_id)
            print(result)

if __name__ == "__main__":
    main() # Versi 85 awal