package ga

import (
	"fmt"
	"github.com/Morenim/gom-opencl/bitset"
	"math/rand"
)

// Solution is a bitstring solution to a maximization optimization problem.
type Solution struct {
	Fitness float64
	Bits    bitset.BitSet
}

func (s Solution) String() string {
	return fmt.Sprintf("%v %v", s.Bits, s.Fitness)
}

func randomSolution(length int) Solution {
	var s Solution
	s.Bits = bitset.New(length)
	for i := 0; i < length; i++ {
		if rand.Float32() > 0.5 {
			s.Bits.Set(i)
		}
	}
	//s.Objective, s.Constraint = evaluate(s.Bits)
	return s
}
