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

__kernel void gom (global read_only uint *population, global read_only uint *fos, global write_only uint *offspring)
{
  int gid = get_global_id (0);
  
  uint solution = population[gid];
  uint fitness = deceptive(population[gid]);

  uint fosPtr = 0;

  while (fos[fosPtr] != 0)
  {
    uint size = fos[fosPtr];
    fosPtr++;

    // Select a 'random' donor.
    uint rand = (gid * 13 + 7 * fosPtr) % 32;

    // Create mask from FOS.
    uint mask = 0;
    for (uint i = 0; i < size; i++)
      mask |= 1 << fos[fosPtr + i];

    // Copy donor bits into the solution.
    uint clone = (solution & ~mask) | (population[rand] & mask);

    uint newFitness = deceptive(clone);

    if (newFitness >= fitness)
    {
      fitness = newFitness;
      solution = clone;
    }

    fosPtr += size;
  }

  offspring[gid] = solution;
}
