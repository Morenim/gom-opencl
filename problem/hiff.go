package problem

import (
	"github.com/Morenim/gom-opencl/bitset"
)

type HIFF int

func (_ HIFF) Evaluate(bits bitset.BitSet) (fitness float64, optimal bool) {

	blockSize := 2
	optimal = true

	for blockSize <= bits.Len() {
		for i := 0; i < bits.Len(); i += blockSize {
			first := bits.Has(i)
			same := true
			for j := i + 1; j < i+blockSize; j++ {
				if bits.Has(j) != first {
					same = false
					optimal = false
					break
				}
			}
			if same {
				fitness += float64(blockSize)
			}
		}
		blockSize *= 2
	}

	return
}
