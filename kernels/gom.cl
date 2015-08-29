// Implements the core functionality of the GOMEA algorithm.
kernel void gom(global uint *population, uint population_size, global uint *fos, global write_only uint *offspring)
{
  int gid = get_global_id (0);
  
  uint solution = population[gid];
  uint fitness = evaluate(solution);
  uint4 rng_state = rng(gid);

  uint fos_size = fos[0];
  uint fos_ptr = 1;

  for (uint fos_index = 0; fos_index < fos_size; ++fos_index)
  {
    uint rand = randrange(&rng_state, 0, population_size - 1);
    uint numMasks = fos[fos_ptr];
    uint clone = 0;
    ++fos_ptr;

    for (uint j = 0; j < numMasks; j++)
    {
      uint mask_index = fos[fos_ptr];
      uint mask = fos[fos_ptr + 1];
      clone = (solution & ~mask) | (population[rand] & mask);
      fos_ptr += 2;
    }

    uint newFitness = evaluate(clone);

    if (newFitness >= fitness)
    {
      fitness = newFitness;
      solution = clone;
    }
  }

  offspring[gid] = solution;
}
