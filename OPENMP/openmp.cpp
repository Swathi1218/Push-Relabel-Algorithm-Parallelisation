#include <iostream>
#include <vector>
#include <deque>
#include <climits>
#include <algorithm>
#include <fstream>
#include <queue>
#include <omp.h>
using namespace std;
struct Edge {
    int from, to;
    long long capacity, flow;
    Edge(int u, int v, long long cap) : from(u), to(v), capacity(cap), flow(0) {}
};
class MaxFlow {
    int V;
    vector<vector<int>> adj;
    vector<Edge> edges;
    vector<int> height, excess;
    deque<int> active;
    vector<bool> inActive;
    int opCount, globalUpdateThreshold;
public:
    MaxFlow(int vertices)
        : V(vertices), adj(vertices), height(vertices, 0), excess(vertices, 0),
          inActive(vertices, false), opCount(0), globalUpdateThreshold(vertices) {}
    void addEdge(int u, int v, long long capacity) {
        edges.emplace_back(u, v, capacity);
        edges.emplace_back(v, u, 0);
        adj[u].push_back(edges.size() - 2);
        adj[v].push_back(edges.size() - 1);
    }
    void push(int e) {
        Edge &edge = edges[e];
        long long pushFlow = min((long long)excess[edge.from], edge.capacity - edge.flow);
        if (pushFlow > 0 && height[edge.from] == height[edge.to] + 1) {
            #pragma omp atomic
            edge.flow += pushFlow;
            #pragma omp atomic
            edges[e ^ 1].flow -= pushFlow;
            #pragma omp atomic
            excess[edge.from] -= pushFlow;
            #pragma omp atomic
            excess[edge.to] += pushFlow;
            if (edge.to != 0 && edge.to != V - 1 && excess[edge.to] > 0 && !inActive[edge.to]) {
                #pragma omp critical
                {
                    active.push_back(edge.to);
                    inActive[edge.to] = true;
                }
            }
        }
    }
    void relabel(int u) {
        int minHeight = INT_MAX;
        for (int e : adj[u]) {
            if (edges[e].capacity > edges[e].flow)
                minHeight = min(minHeight, height[edges[e].to]);
        }
        if (minHeight < INT_MAX)
            height[u] = minHeight + 1;
    }

    void discharge(int u, int sink) {
        bool workDone = true;  // Track if we did any push operations
        while (excess[u] > 0 && workDone) {
            workDone = false;
            #pragma omp parallel for
            for (int i = 0; i < adj[u].size(); ++i) {
                int e = adj[u][i];
                if (edges[e].capacity > edges[e].flow && height[u] == height[edges[e].to] + 1) {
                    push(e);
                    workDone = true; // Mark work done
                }
            }
            if (!workDone) {
                #pragma omp single
                relabel(u);
            }
        }
    }
    long long maxFlow(int source, int sink) {
        if (source < 0 || sink < 0 || source >= V || sink >= V) {
            cerr << "Invalid source or sink." << endl;
            return 0;
        }
        height[source] = V;
        excess[source] = 0;
        #pragma omp parallel for
        for (int i = 0; i < adj[source].size(); ++i) {
            int e = adj[source][i];
            Edge &edge = edges[e];
            if (edge.capacity > 0) {
                long long pushFlow = edge.capacity;
                edge.flow = pushFlow;
                edges[e ^ 1].flow = -pushFlow;
                excess[edge.to] += pushFlow;
                excess[source] -= pushFlow;
                #pragma omp critical
                {
                    if (edge.to != source && edge.to != sink && !inActive[edge.to]) {
                        active.push_back(edge.to);
                        inActive[edge.to] = true;
                    }
                }
            }
        }
        while (!active.empty()) {
            int u;
            #pragma omp critical
            {
                u = active.front();
                active.pop_front();
                inActive[u] = false;
            }
            if (u != source && u != sink)
                discharge(u, sink);
        }
        long long totalFlow = 0;
        for (int e : adj[source]) {
            totalFlow += edges[e].flow;
        }
        return totalFlow;
    }
};


int main() {
    ifstream infile("input.txt");
    if (!infile) {
        cerr << "Error opening file." << endl;
        return 1;
    }
    int V, source, sink;
    infile >> V >> source >> sink;
    MaxFlow maxFlow(V);
    int u, v;
    long long capacity;
    while (infile >> u >> v >> capacity) {
        maxFlow.addEdge(u, v, capacity);
    }
    infile.close();
    cout << "Threads, Max Flow, Execution Time (s)" << endl;
    vector<int> threadCounts = {1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64};
    for (int threads : threadCounts) {
        omp_set_num_threads(threads);
        double start = omp_get_wtime();
        long long result = maxFlow.maxFlow(source, sink);
        double end = omp_get_wtime();
        cout << threads << ", " << result << ", " << (end - start) << endl;
    }
    return 0;
}
