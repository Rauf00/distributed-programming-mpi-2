#include <iostream>
#include <cstdio>
#include <mpi.h>
#include "core/utils.h"
#include "core/graph.h"

long countTriangles(uintV *array1, uintE len1, uintV *array2, uintE len2,
                     uintV u, uintV v) {
  uintE i = 0, j = 0; // indexes for array1 and array2
  long count = 0;

  if (u == v)
    return count;

  while ((i < len1) && (j < len2)) {
    if (array1[i] == array2[j]) {
      if ((array1[i] != u) && (array1[i] != v)) {
        count++;
      } else {
        // triangle with self-referential edge -> ignore
      }
      i++;
      j++;
    } else if (array1[i] < array2[j]) {
      i++;
    } else {
      j++;
    }
  }
  return count;
}

void triangleCountGather(Graph &g, int world_rank, int world_size){
  timer exec_timer;
  timer communication_timer;

  exec_timer.start();

  double exec_time = 0.0;
  double communication_time = 0.0;

  uintV n = g.n_;
  uintE m = g.m_;

  // Edge decomposition strategy for a graph with n vertices and m edges for P processes
  uintV start_vertex = 0;
  uintV end_vertex = 0;
  for(int i=0; i < world_size; i++){
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
  // Each process will work on vertices [start_vertex, end_vertex).

  // Each process calculates its local triangle count
  long local_count = 0;
  long local_edge_count = 0;

  for (uintV u = start_vertex; u < end_vertex; u++) {
    uintE out_degree = g.vertices_[u].getOutDegree();
    local_edge_count += out_degree;
    for (uintE i = 0; i < out_degree; i++) {
      uintV v = g.vertices_[u].getOutNeighbor(i);
      local_count += countTriangles(g.vertices_[u].getInNeighbors(),
                                       g.vertices_[u].getInDegree(),
                                       g.vertices_[v].getOutNeighbors(),
                                       g.vertices_[v].getOutDegree(), u, v);
    }
  }

  // --- synchronization phase start ---

  communication_timer.start();

  // if P is root process
  long *local_counts = NULL;
  if(world_rank == 0){
    local_counts = new long[world_size];
  }

  MPI_Gather(&local_count, 1, MPI_LONG, local_counts, 1, MPI_LONG, 0, MPI_COMM_WORLD);

  long global_count = 0;
  if(world_rank == 0){
    for(int i = 0; i < world_size; i++){
      global_count += local_counts[i];
    }
  }

  communication_time = communication_timer.stop();

  // --- synchronization phase end -----
  // if P is root process
  if(world_rank == 0){
      delete[] local_counts;
      exec_time = exec_timer.stop();

      // print process statistics and other results
      std::printf("rank, edges, triangle_count, communication_time\n");

      std::printf("%d, %ld, %ld, %f\n", world_rank, local_edge_count, local_count, communication_time);

      std::printf("Number of triangles : %ld\n", global_count);
      std::printf("Number of unique triangles : %ld\n", global_count / 3);
      std::printf("Time taken (in seconds) : %f\n", exec_time);
  }
  else{
    // print process statistics
    std::printf("%d, %ld, %ld, %f\n", world_rank, local_edge_count, local_count, communication_time);
  }
}

void triangleCountReduce(Graph &g, int world_rank, int world_size){
  timer exec_timer;
  timer communication_timer;

  exec_timer.start();

  double exec_time = 0.0;
  double communication_time = 0.0;

  uintV n = g.n_;
  uintE m = g.m_;

  // Edge decomposition strategy for a graph with n vertices and m edges for P processes
  uintV start_vertex = 0;
  uintV end_vertex = 0;
  for(int i=0; i < world_size; i++){
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
  // Each process will work on vertices [start_vertex, end_vertex).

  // Each process calculates its local triangle count
  long local_count = 0;
  long local_edge_count = 0;

  for (uintV u = start_vertex; u < end_vertex; u++) {
    uintE out_degree = g.vertices_[u].getOutDegree();
    local_edge_count += out_degree;
    for (uintE i = 0; i < out_degree; i++) {
      uintV v = g.vertices_[u].getOutNeighbor(i);
      local_count += countTriangles(g.vertices_[u].getInNeighbors(),
                                       g.vertices_[u].getInDegree(),
                                       g.vertices_[v].getOutNeighbors(),
                                       g.vertices_[v].getOutDegree(), u, v);
    }
  }

  // --- synchronization phase start ---

  communication_timer.start();

  long global_count;
  MPI_Reduce(&local_count, &global_count, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  communication_time = communication_timer.stop();

  // --- synchronization phase end -----
  // if P is root process
  if(world_rank == 0){
      exec_time = exec_timer.stop();

      // print process statistics and other results
      std::printf("rank, edges, triangle_count, communication_time\n");

      std::printf("%d, %ld, %ld, %f\n", world_rank, local_edge_count, local_count, communication_time);

      std::printf("Number of triangles : %ld\n", global_count);
      std::printf("Number of unique triangles : %ld\n", global_count / 3);
      std::printf("Time taken (in seconds) : %f\n", exec_time);
  }
  else{
    // print process statistics
    std::printf("%d, %ld, %ld, %f\n", world_rank, local_edge_count, local_count, communication_time);
  }
}


int main(int argc, char *argv[])
{
    cxxopts::Options options("triangle_counting_serial", "Count the number of triangles using serial and parallel execution");
    options.add_options("custom", {
                                      {"strategy", "Strategy to be used", cxxopts::value<uint>()->default_value(DEFAULT_STRATEGY)},
                                      {"inputFile", "Input graph file path", cxxopts::value<std::string>()->default_value("/scratch/input_graphs/roadNet-CA")},
                                  });

    auto cl_options = options.parse(argc, argv);
    uint strategy = cl_options["strategy"].as<uint>();
    std::string input_file_path = cl_options["inputFile"].as<std::string>();

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    // Find out rank, size
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(world_rank == 0){
      // Get the world size and print it out here
      std::printf("World size : %d\n", world_size);
      std::printf("Communication strategy : %d\n", strategy);
    }

    Graph g;
    g.readGraphFromBinary<int>(input_file_path);

    switch (strategy) {
      case 1:
        triangleCountGather(g, world_rank, world_size);
        break;
      case 2:
        triangleCountReduce(g, world_rank, world_size);
        break;
      default:
        break;
    }

    MPI_Finalize();

    return 0;
}

