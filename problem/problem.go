package problem

import (
	"github.com/Morenim/gom-opencl/bitset"
)

type Problem interface {
	Evaluate(bits bitset.BitSet) (float64, bool)
}
