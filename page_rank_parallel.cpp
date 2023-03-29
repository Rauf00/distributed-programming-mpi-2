#include <iostream>
#include <cstdio>
#include <mpi.h>
#include "core/utils.h"
#include "core/graph.h"

#ifdef USE_INT
#define INIT_PAGE_RANK 100000
#define EPSILON 1000
#define PAGE_RANK(x) (15000 + (5 * x) / 6)
#define CHANGE_IN_PAGE_RANK(x, y) std::abs(x - y)
#define PAGERANK_MPI_TYPE MPI_LONG
#define PR_FMT "%ld"
typedef int64_t PageRankType;
#else
#define INIT_PAGE_RANK 1.0
#define EPSILON 0.01
#define DAMPING 0.85
#define PAGE_RANK(x) (1 - DAMPING + DAMPING * x)
#define CHANGE_IN_PAGE_RANK(x, y) std::fabs(x - y)
#define PAGERANK_MPI_TYPE MPI_FLOAT
#define PR_FMT "%f"
typedef float PageRankType;
#endif

void pageRankS1(Graph &g, int max_iters, int world_rank, int world_size){
    timer exec_timer;
    timer communication_timer;
    exec_timer.start();

    double exec_time = 0.0;
    double communication_time = 0.0;

    uintV n = g.n_;
    uintE m = g.m_;

    PageRankType *pr_curr = new PageRankType[n];
    PageRankType *pr_next = new PageRankType[n];

    for (uintV i = 0; i < n; i++) {
        pr_curr[i] = INIT_PAGE_RANK;
        pr_next[i] = 0.0;
    }

    // Edge decomposition strategy for a graph with n vertices and m edges for P processes
    uintV start_vertex = 0;
    uintV end_vertex = 0;
    for(int i = 0; i < world_size; i++){
        start_vertex = end_vertex;
        long count = 0;
        while (end_vertex < n)
        {
            // add vertices until we reach m/P edges.
            count += g.vertices_[end_vertex].getOutDegree();
            end_vertex += 1;
            if (count >= m / world_size)
                break;
        }
        if(i == world_rank)
            break;
    }

    uintV *end_vertices = new uintV[world_size];
    if(world_rank == 0){
        uintV start_vertex = 0;
        uintV end_vertex = 0;
        for(int i = 0; i < world_size; i++){
            start_vertex = end_vertex;
            long count = 0;
            while (end_vertex < n)
            {
                // add vertices until we reach m/P edges.
                count += g.vertices_[end_vertex].getOutDegree();
                end_vertex += 1;
                if (count >= m / world_size)
                    break;
            }
            end_vertices[i] = end_vertex;
        }
    }
    
    // Each process will work on vertices [start_vertex, end_vertex).

    long local_vertex_count = 0;
    long local_edge_count = 0;

    for (int iter = 0; iter < max_iters; iter++) {
        for (uintV u = start_vertex; u < end_vertex; u++) {
            uintE out_degree = g.vertices_[u].getOutDegree();
            local_edge_count += out_degree;
            for (uintE i = 0; i < out_degree; i++) {
                uintV v = g.vertices_[u].getOutNeighbor(i);
                pr_next[v] += pr_curr[u] / out_degree;
            }
        }

        communication_timer.start();

        // if P is root process
        PageRankType *global_pr_next = NULL;
        if(world_rank == 0){
            global_pr_next = new PageRankType[n];
        }
        MPI_Reduce(pr_next, global_pr_next, n, PAGERANK_MPI_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);

        // The root process sends the aggregated next_page_rank value of each vertex to its appropriate process
        int *sendcounts = NULL;
        int *displs = NULL;

        if(world_rank == 0){
            for(int i = 0; i < n; i++){
                pr_next[i] = global_pr_next[i];
            }
            sendcounts = (int*)malloc(sizeof(int) * world_size);
            displs = (int*)malloc(sizeof(int) * world_size);
            uintV sub_start = 0;
            uintV sub_end = end_vertices[0];
            // displs[0] = sub_start;
            // sendcounts[0] = sub_end - sub_start;
            for(int i = 0; i < world_size; i++){
                //std::printf("sub_start: %d, sub_end: %d, sub_len: %d", sub_start, sub_end, sub_end - sub_start);
                displs[i] = sub_start;
                sendcounts[i] = sub_end - sub_start;
                if(i + 1 == world_size) { 
                    break;
                }
                sub_start = end_vertices[i];
                sub_end = end_vertices[i + 1];
            }
        } 

        uintV len = end_vertex - start_vertex;
        PageRankType *sub_pr_next = new PageRankType[len];
        MPI_Scatterv(pr_next, sendcounts, displs, PAGERANK_MPI_TYPE, sub_pr_next, len, PAGERANK_MPI_TYPE, 0, MPI_COMM_WORLD);
        uintV j = 0;
        for (uintV u = start_vertex; u < end_vertex; u++) {
            pr_next[u] = sub_pr_next[j];
            j++;
        }
        communication_time += communication_timer.stop();
        
        for (uintV v = start_vertex; v < end_vertex; v++) {
            local_vertex_count++;
            pr_next[v] = PAGE_RANK(pr_next[v]);
            pr_curr[v] = pr_next[v];
        }

        //Reset next_page_rank[v] to 0 for all vertices
        for (uintV v = 0; v < n; v++) {
            pr_next[v] = 0.0;
        }

        if(world_rank == 0){
            free(sendcounts);
            free(displs);
            delete[] global_pr_next;
        }
        delete[] sub_pr_next;
    }

    PageRankType local_sum = 0.0;
    for (uintV v = start_vertex; v < end_vertex; v++) {  // Loop 3
        local_sum += pr_curr[v];
    }

    delete[] pr_curr;
    delete[] pr_next;
    if(world_rank == 0){
        delete[] end_vertices;
    }

    PageRankType global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, PAGERANK_MPI_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);

    // --- synchronization phase 2 start ---
    if(world_rank == 0){
        exec_time = exec_timer.stop();
        
        std::printf("rank, num_edges, communication_time\n");

        std::printf("%d, %ld, %f\n", world_rank, local_edge_count, communication_time);

        std::printf("Sum of page rank : " PR_FMT "\n", global_sum);
        std::printf("Time taken (in seconds) : %f\n", exec_time);
    }
    else{
        // print process statistics.
        std::printf("%d, %ld, %f\n", world_rank, local_edge_count, communication_time);
    }
    // --- synchronization phase 2 end ---
}

void pageRankS2(Graph &g, int max_iters, int world_rank, int world_size){
    timer exec_timer;
    timer communication_timer;
    exec_timer.start();

    double exec_time = 0.0;
    double communication_time = 0.0;

    uintV n = g.n_;
    uintE m = g.m_;

    PageRankType *pr_curr = new PageRankType[n];
    PageRankType *pr_next = new PageRankType[n];

    for (uintV i = 0; i < n; i++) {
        pr_curr[i] = INIT_PAGE_RANK;
        pr_next[i] = 0.0;
    }

    // Edge decomposition strategy for a graph with n vertices and m edges for P processes
    uintV start_vertex = 0;
    uintV end_vertex = 0;
    for(int i = 0; i < world_size; i++){
        start_vertex = end_vertex;
        long count = 0;
        while (end_vertex < n)
        {
            // add vertices until we reach m/P edges.
            count += g.vertices_[end_vertex].getOutDegree();
            end_vertex += 1;
            if (count >= m / world_size)
                break;
        }
        if(i == world_rank)
            break;
    }

    uintV *end_vertices = new uintV[world_size];
    if(true){
        uintV start_vertex = 0;
        uintV end_vertex = 0;
        for(int i = 0; i < world_size; i++){
            start_vertex = end_vertex;
            long count = 0;
            while (end_vertex < n)
            {
                // add vertices until we reach m/P edges.
                count += g.vertices_[end_vertex].getOutDegree();
                end_vertex += 1;
                if (count >= m / world_size)
                    break;
            }
            end_vertices[i] = end_vertex;
        }
    }
    
    // Each process will work on vertices [start_vertex, end_vertex).

    long local_vertex_count = 0;
    long local_edge_count = 0;

    for (int iter = 0; iter < max_iters; iter++) {
        for (uintV u = start_vertex; u < end_vertex; u++) {
            uintE out_degree = g.vertices_[u].getOutDegree();
            local_edge_count += out_degree;
            for (uintE i = 0; i < out_degree; i++) {
                uintV v = g.vertices_[u].getOutNeighbor(i);
                pr_next[v] += pr_curr[u] / out_degree;
            }
        }

        communication_timer.start();

        // if P is root process
        uintV sub_start = 0;
        uintV sub_end = end_vertices[0];

        uintV len = sub_end - sub_start;
        PageRankType *global_sub_pr_next = new PageRankType[n];
        PageRankType *sub_pr_next = new PageRankType[n];

        for(int i = 0; i < world_size; i++){
            int j = 0;
            std::copy(&pr_next[sub_start], &pr_next[sub_end], sub_pr_next);
            
            MPI_Reduce(sub_pr_next, global_sub_pr_next, len, PAGERANK_MPI_TYPE, MPI_SUM, i, MPI_COMM_WORLD);

            int k = 0;
            for(uintV i = sub_start; i < sub_end; i++){
                pr_next[i] = global_sub_pr_next[k];
                k++;
            }
            if(i + 1 == world_size){
                break;
            }
            sub_start = end_vertices[i];
            sub_end = end_vertices[i + 1];
            len = sub_end - sub_start;
        }
        
        delete[] global_sub_pr_next;
        delete[] sub_pr_next;
        communication_time += communication_timer.stop();
        
        for (uintV v = start_vertex; v < end_vertex; v++) {
            local_vertex_count++;
            pr_next[v] = PAGE_RANK(pr_next[v]);
            pr_curr[v] = pr_next[v];
        }

        //Reset next_page_rank[v] to 0 for all vertices
        for (uintV v = 0; v < n; v++) {
            pr_next[v] = 0.0;
        }
    }

    PageRankType local_sum = 0.0;
    for (uintV v = start_vertex; v < end_vertex; v++) {  // Loop 3
        local_sum += pr_curr[v];
    }

    PageRankType global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, PAGERANK_MPI_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);

    delete[] pr_curr;
    delete[] pr_next;
    delete[] end_vertices;

    // --- synchronization phase 2 start ---
    if(world_rank == 0){
        exec_time = exec_timer.stop();
        
        std::printf("rank, num_edges, communication_time\n");

        std::printf("%d, %ld, %f\n", world_rank, local_edge_count, communication_time);

        std::printf("Sum of page rank : " PR_FMT "\n", global_sum);
        std::printf("Time taken (in seconds) : %f\n", exec_time);
    }
    else{
        // print process statistics.
        std::printf("%d, %ld, %f\n", world_rank, local_edge_count, communication_time);
    }
    // --- synchronization phase 2 end ---
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("page_rank_push", "Calculate page_rank using serial and parallel execution");
    options.add_options("", {
                                {"nIterations", "Maximum number of iterations", cxxopts::value<uint>()->default_value(DEFAULT_MAX_ITER)},
                                {"strategy", "Strategy to be used", cxxopts::value<uint>()->default_value(DEFAULT_STRATEGY)},
                                {"inputFile", "Input graph file path", cxxopts::value<std::string>()->default_value("/scratch/input_graphs/roadNet-CA")},
                            });

    auto cl_options = options.parse(argc, argv);
    uint strategy = cl_options["strategy"].as<uint>();
    uint max_iterations = cl_options["nIterations"].as<uint>();
    std::string input_file_path = cl_options["inputFile"].as<std::string>();

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    // Find out rank, size
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(world_rank == 0){
        #ifdef USE_INT
            std::printf("Using INT\n");
        #else
            std::printf("Using FLOAT\n");
        #endif
        // Get the world size and print it out here
        std::printf("World size : %d\n", world_size);
        std::printf("Communication strategy : %d\n", strategy);
        std::printf("Iterations : %d\n", max_iterations);
    }

    Graph g;
    g.readGraphFromBinary<int>(input_file_path);

    switch (strategy) {
      case 1:
        pageRankS1(g, max_iterations, world_rank, world_size);
        break;
      case 2:
        pageRankS2(g, max_iterations, world_rank, world_size);
        break;
      default:
        break;
    }

    return 0;
}
