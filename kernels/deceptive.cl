__constant int problemLength;
__constant int populationSize;
__constant int fosSize;

uint deceptive (__read_only uint solution)
{
  uint fitness = 0;
  uint k = 4;
  uint numTraps = 32 >> 2;

  uint z = 1;
  for (uint i = 0; i < numTraps; i++)
  {
    uint t = 0;
    for (uint j = 0; j < k; j++)
    {
      t += (z & solution) ? 1 : 0;
      z <<= 1;
    }

    if(t == k) 
    {
      fitness += k;
    }
    else
    {
      fitness += k - t - 1;
    }
  }

  return fitness;
}

kernel void gom (global uint *population, uint population_size, global uint *fos, global write_only uint *offspring)
{
  int gid = get_global_id (0);
  
  uint solution = population[gid];
  uint fitness = deceptive(solution);

  uint fosPtr = 0;

  while (fos[fosPtr] != 0)
  {
    uint subsetSize = fos[fosPtr];
    fosPtr++;

    // Select a 'random' donor.
    uint rand = (gid * 13 + 7 * fosPtr) % population_size;

    // Create mask from FOS.
    uint mask = 0;
    for (uint i = 0; i < subsetSize; i++)
      mask |= 1 << fos[fosPtr + i];

    // Copy donor bits into the solution.
    uint clone = (solution & ~mask) | (population[rand] & mask);

    uint newFitness = deceptive(clone);

    if (newFitness >= fitness)
    {
      fitness = newFitness;
      solution = clone;
    }

    fosPtr += subsetSize;
  }

  offspring[gid] = solution;
}
