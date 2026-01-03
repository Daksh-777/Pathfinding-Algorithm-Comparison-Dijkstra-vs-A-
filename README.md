# Pathfinding-Algorithm-Comparison-Dijkstra-vs-A-
Investigation of the question -> How does the pathfinding efficiency of Dijkstra's algorithm compare to that of the A* algorithm in terms of time complexity on different types of graphs?

## 📊 Key Findings

- **On sparse graphs**: Dijkstra’s often outperforms A\* at small-to-medium scales due to predictable expansion; A\* shows higher variance due to heuristic sensitivity.
- **On dense graphs**: A\* generally outperforms Dijkstra’s for |V| < 1,500 by leveraging Euclidean distance heuristics to prune search space.
- Both algorithms follow **O((V + E) log V)** complexity with priority queues, but **real-world runtime differs significantly** based on graph structure and heuristic quality.

> 💡 **Conclusion**: A\* is superior in structured, dense environments (e.g., game maps, city navigation), while Dijkstra’s remains robust in unstructured or sparse networks.

---

## 🛠️ Technologies Used

- **Python 3.11**
- `networkx` — Graph generation and representation
- `matplotlib` — Data visualization
- `numpy` — Numerical operations
- `time` — High-precision timing (`perf_counter_ns`)
- **Custom implementations** of Dijkstra’s and A\* (no external solvers)

---

## 🧪 Methodology

- Generated **directed weighted graphs** with |V| ∈ {100, 200, ..., 1000}
- Created **sparse graphs** (|E| ≈ 2×|V|) and **dense graphs** (|E| ≈ 0.5×|V|²)
- Edge weights: `U(1, 10)`
- A\* heuristic: **Euclidean distance** between node coordinates
- Measured:
  - Execution time (nanoseconds)
  - Node expansion count
- Used **non-linear regression** (quadratic & power models) to analyze scaling behavior

---
