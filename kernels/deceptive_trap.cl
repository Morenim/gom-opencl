uint trap(read_only uint solution, int offset)
{
  uint k = 4;
  uint fitness = 0;

  uint t = 0;
  uint z = 1 << offset;
  for (uint j = 0; j < k; j++)
  {
    t += (z & solution) ? 1 : 0;
    z <<= 1;
  }

  if (t == k)
    fitness += k;
  else
    fitness += k - t - 1;

  return fitness;
}

// Implements the evaluation function for the
// concatenated deceptive trap function.
uint evaluate(read_only global uint *solution, read_only uint solution_length)
{
  uint fitness = 0;
  uint k = 4;

  for (uint i = 0; i < solution_length; i += k)
    fitness += trap(solution[i >> 5], i & 31);

  return fitness;
}
