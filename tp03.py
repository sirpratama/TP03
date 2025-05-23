from collections import deque
import heapq

class GalacticGraph:
    def __init__(self, n, planet_types_str):
        # Direct character access is faster than split()
        types = planet_types_str.strip()
        self.is_prime = [False] * n
        self.adj = [[] for _ in range(n)]
        self.edges = []
        
        # Parse planet types more efficiently
        idx = 0
        for i in range(n):
            if idx < len(types):
                if types[idx] == 'P':
                    self.is_prime[i] = True
                # Skip to next non-space character
                idx += 1
                while idx < len(types) and types[idx] == ' ':
                    idx += 1
        
        self.n = n
        self.prime_indices = [i for i in range(n) if self.is_prime[i]]

    def add_edge(self, u, v, e):
        u -= 1
        v -= 1
        self.adj[u].append((v, e))
        self.adj[v].append((u, e))
        self.edges.append((e, u, v))

def mission_a(graph, start, energy):
    # Optimized BFS with early exit conditions
    if not graph.adj[start]:
        return 1 if graph.is_prime[start] else 0
    
    visited = [False] * graph.n
    visited[start] = True
    queue = deque([start])
    prime_count = 0
    
    # Count initial planet if prime
    if graph.is_prime[start]:
        prime_count += 1
    
    while queue:
        u = queue.popleft()
        
        # Process all neighbors at once
        for v, e in graph.adj[u]:
            if not visited[v] and e <= energy:
                visited[v] = True
                if graph.is_prime[v]:
                    prime_count += 1
                queue.append(v)
    
    return prime_count

def mission_b(graph, start):
    # Special case: starting planet is prime
    if graph.is_prime[start]:
        return 0
    
    # Modified Dijkstra with early termination
    dist = [float('inf')] * graph.n
    dist[start] = 0
    heap = [(0, start)]
    
    while heap:
        d, u = heapq.heappop(heap)
        
        if d > dist[u]:
            continue
            
        # Check if we reached a prime planet
        if graph.is_prime[u]:
            return d
        
        for v, e in graph.adj[u]:
            new_dist = max(d, e)
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    
    return -1

def mission_c(graph):
    # Kruskal's algorithm - already efficient
    edges = sorted(graph.edges)
    parent = list(range(graph.n))
    rank = [0] * graph.n
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True
    
    total = 0
    edges_used = 0
    
    for e, u, v in edges:
        if union(u, v):
            total += e
            edges_used += 1
            if edges_used == graph.n - 1:
                break
    
    return total

def dijkstra(graph, start):
    dist = [float('inf')] * graph.n
    dist[start] = 0
    heap = [(0, start)]
    
    while heap:
        d, u = heapq.heappop(heap)
        
        if d > dist[u]:
            continue
        
        for v, e in graph.adj[u]:
            if d + e < dist[v]:
                dist[v] = d + e
                heapq.heappush(heap, (d + e, v))
    
    return dist

def mission_d(graph, start, end):
    # Direct path
    dist_from_start = dijkstra(graph, start)
    min_cost = dist_from_start[end]
    
    # With quantum leap - optimized computation
    if graph.prime_indices:
        dist_to_end = dijkstra(graph, end)
        
        # Find minimum cost using quantum leap
        min_to_prime = min(dist_from_start[p] for p in graph.prime_indices)
        min_from_prime = min(dist_to_end[p] for p in graph.prime_indices)
        
        if min_to_prime < float('inf') and min_from_prime < float('inf'):
            min_cost = min(min_cost, min_to_prime + min_from_prime)
    
    return min_cost

def main():
    # Fast input reading
    import sys
    input = sys.stdin.readline
    
    n, m = map(int, input().split())
    planet_types = input()
    
    graph = GalacticGraph(n, planet_types)
    
    for _ in range(m):
        u, v, e = map(int, input().split())
        graph.add_edge(u, v, e)
    
    q = int(input())
    
    for _ in range(q):
        cmd = input().split()
        
        if cmd[0] == 'A':
            print(mission_a(graph, 0, int(cmd[1])))
        elif cmd[0] == 'B':
            s, t = int(cmd[1]) - 1, int(cmd[2]) - 1
            if graph.is_prime[t]:
                print(0)
            else:
                print(mission_b(graph, s))
        elif cmd[0] == 'C':
            print(mission_c(graph))
        elif cmd[0] == 'D':
            g, d = int(cmd[1]) - 1, int(cmd[2]) - 1
            print(mission_d(graph, g, d))

if __name__ == "__main__":
    main()