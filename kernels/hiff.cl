// Implements the evaluation function for the
// Hierarchical If and only If function.
uint evaluate(read_only global uint *solution, read_only uint solution_length)
{
  uint fitness = 0;
  uint k = 4;
  uint block_size = 2;

  while (block_size <= solution_length)
  {
    for (uint i = 0; i < solution_length; i += block_size)
    {
      uint t = 0;

      for (uint j = i; j < i + block_size; j++)
        t += solution[j >> 5] & (1 << (j & 31)) ? 1 : 0;

      if (t == block_size || t == 0)
        fitness += block_size;
    }
    
    block_size *= 2;
  }

  return fitness;
}
