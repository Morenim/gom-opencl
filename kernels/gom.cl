int ints_per_solution(int solution_length)
{
  return 1 + ((solution_length - 1) >> 5);
}

void copyMasks(uint num_masks, global uint *fos, global uint *src, global uint* dest) 
{

}

// Implements the core functionality of the GOMEA algorithm.
kernel void gom(global uint *population, const uint population_size, const uint solution_length, global uint *clones, global uint *fos, global write_only uint *offspring)
{
  int gid = get_global_id (0);
  uint4 rng_state = rng(gid);
  uint num_ints_solution = ints_per_solution(solution_length);
  uint intdex = (gid * num_ints_solution);

  // Initialize the clone / offspring memory.
  for (uint i = intdex; i < intdex + num_ints_solution; i++)
  {
    clones[i] = population[i];
    offspring[i] = population[i];
  }

  uint fitness = evaluate(&offspring[intdex], solution_length);

  uint fos_size = fos[0];
  uint fos_ptr = 1;

  for (uint fos_index = 0; fos_index < fos_size; ++fos_index)
  {
    uint rand = randrange(&rng_state, 0, population_size - 1);
    uint num_masks = fos[fos_ptr];

    for (uint j = 0; j < num_masks; j++)
    {
      uint mask_index = fos[fos_ptr + 2 * j + 1];
      uint mask = fos[fos_ptr + 2 * j + 2];
      uint changes = population[num_ints_solution * rand] & mask;
      clones[intdex + mask_index] = (offspring[intdex + mask_index] & ~mask) | changes;
    }

    uint newFitness = evaluate(clones + intdex, solution_length);

    if (newFitness >= fitness)
    {
      for (uint j = 0; j < num_masks; j++)
      {
        uint mask_index = intdex + fos[fos_ptr + 2 * j + 1];
        offspring[mask_index] = clones[mask_index];
      }
      fitness = newFitness;
    }
    else
    {
      for (uint j = 0; j < num_masks; j++)
      {
        uint mask_index = intdex + fos[fos_ptr + 2 * j + 1];
        clones[mask_index] = offspring[mask_index];
      }
    }

    fos_ptr += 2 * num_masks + 1;
  }
}
