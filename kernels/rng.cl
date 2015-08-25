// Implementation of the Tyche-i RNG by Samuel Neves and Filipe Araujo.

// Returns a bit rotation of the integer x to the left by n bits.
uint rotl32(uint x, int n)
{
  return (x << n) | (x >> (32-n));
}

// Returns a random number in the range [0, 2^32-1]. 
uint randint(uint4 *state)
{
  uint4 s = *state;
  s.yw = (uint2)(rotl32(s.y, 25) ^ s.z, rotl32(s.w, 24) ^ s.x);
  s.xz -= s.yw;
  s.yw = (uint2)(rotl32(s.y, 20) ^ s.z, rotl32(s.w, 16) ^ s.x);
  s.xz -= s.yw;
  *state = s;
  return s.y;
}

// Returns a random number in the range [x, y].
uint randrange(uint4 *state, int x, int y)
{
  return x + (uint)((ulong)(randint(state) * (ulong)(y - x + 1)) >> 32);
}

// Returns a random number in the range [0, 1].
float rand()
{
  return randint() * (1f / 4294967296f);
}

uint4 rng(ulong seed) 
{
  uint4 state = (uint4)(seed << 32, seed & 0xffffffffu, 2654435769u, 1367130551u);

  for (int i = 0; i < 20; ++i)
    randint(&state);

  return state;
}
