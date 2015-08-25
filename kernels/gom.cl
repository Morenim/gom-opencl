// Implements the core functionality of the GOMEA algorithm.
kernel void gom(global uint *population, uint population_size, global uint *fos, global write_only uint *offspring)
{
  int gid = get_global_id (0);
  
  uint solution = population[gid];
  uint fitness = evaluate(solution);
  uint4 rng_state = rng(gid);

  uint fosPtr = 0;

  while (fos[fosPtr] != 0)
  {
    uint subsetSize = fos[fosPtr];
    fosPtr++;

    // Select a 'random' donor.
    //uint rand = (gid * 13 + 7 * fosPtr) % population_size;
    uint rand = randrange(&rng_state, 0, population_size - 1);

    // Create mask from FOS.
    uint mask = 0;
    for (uint i = 0; i < subsetSize; i++)
      mask |= 1 << fos[fosPtr + i];

    // Copy donor bits into the solution.
    uint clone = (solution & ~mask) | (population[rand] & mask);

    uint newFitness = evaluate(clone);

    if (newFitness >= fitness)
    {
      fitness = newFitness;
      solution = clone;
    }

    fosPtr += subsetSize;
  }

  offspring[gid] = solution;
}
