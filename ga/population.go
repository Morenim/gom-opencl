package ga

import (
	"fmt"
)

// Population is a collection of solutions.
type Population struct {
	Solutions []Solution
}

func (pop *Population) Size() int {
	return len(pop.Solutions)
}

func (pop *Population) Length() int {
	return pop.Solutions[0].Bits.Len()
}

// NewPopulation returns an unevaluated population.
func NewPopulation(size, length int) *Population {
	pop := new(Population)
	pop.Solutions = make([]Solution, size)
	for i := 0; i < size; i++ {
		pop.Solutions[i] = randomSolution(length)
		pop.Solutions[i].Fitness = 0.0
	}
	return pop
}

func (pop *Population) String() string {
	return fmt.Sprintf("%v", pop.Solutions)
}
