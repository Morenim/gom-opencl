package problem

import (
	"github.com/Morenim/gom-opencl/bitset"
)

type DeceptiveTrap int

func (dt DeceptiveTrap) Evaluate(bits bitset.BitSet) (fitness float64, optimal bool) {
	k := int(dt)
	optimal = true

	for i := 0; i < bits.Len()/k; i++ {
		t := 0 // number of bits set to 1
		for j := 0; j < k; j++ {
			if bits.Has(i*k + j) {
				t++
			}
		}
		if t == k {
			fitness += float64(t)
		} else {
			fitness += float64(k - t - 1)
			optimal = false
		}
	}
	return
}
