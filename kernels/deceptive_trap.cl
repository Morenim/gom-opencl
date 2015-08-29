// Implements the evaluation function for the
// concatenated deceptive trap function.
uint evaluate(read_only uint solution)
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