# TP03 - Galactic Graph Algorithms

This project implements graph algorithms for planetary mission operations in a galactic setting.

## Features

The program supports four types of missions:

- **Mission A**: Initial Exploration - Count reachable prime planets with limited energy
- **Mission B**: Rescue Operation - Find minimum energy to reach any prime planet
- **Mission C**: Infrastructure Development - Calculate minimum spanning tree energy cost
- **Mission D**: Strategic Planning - Find minimum energy path with optional prime planet stops

## Algorithms Used

- **BFS (Breadth-First Search)** for Mission A
- **Modified Dijkstra's Algorithm** for Mission B (finding minimum bottleneck path)
- **Kruskal's Algorithm with Disjoint Set Union** for Mission C
- **Standard Dijkstra's Algorithm** for Mission D

## Input Format

```
n m
planet_types
u1 v1 e1
u2 v2 e2
...
um vm em
q
mission_commands
```

Where:
- `n` = number of planets
- `m` = number of warp lanes
- `planet_types` = string of 'P' (prime) or 'N' (normal) for each planet
- Each warp lane connects planets `u` and `v` with energy cost `e`
- `q` = number of mission queries

## Mission Commands

- `A energy` - Mission A with given energy limit
- `B s_id t_id` - Mission B from planet s_id, checking if t_id is prime
- `C` - Mission C (MST calculation)
- `D g_id d_id` - Mission D from g_id to d_id

## Usage

```bash
python tp03.py < input.txt
``` 