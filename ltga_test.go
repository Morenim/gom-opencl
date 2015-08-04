package main

import (
	"fmt"
	"github.com/Morenim/gom-opencl/bitset"
	"sort"
	"testing"
)

func IntArrayEquals(a []int, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

func first(bits bitset.BitSet, err error) bitset.BitSet {
	return bits
}

var deceptivePopulation = Population{
	Solutions: []Solution{
		Solution{Fitness: 10, Bits: first(bitset.FromString("1111000011110000"))},
		Solution{Fitness: 10, Bits: first(bitset.FromString("0000111111110000"))},
		Solution{Fitness: 10, Bits: first(bitset.FromString("1111000011111111"))},
		Solution{Fitness: 10, Bits: first(bitset.FromString("1111000011110000"))},
		Solution{Fitness: 10, Bits: first(bitset.FromString("0000000000001111"))},
		Solution{Fitness: 10, Bits: first(bitset.FromString("0000111111110000"))},
		Solution{Fitness: 10, Bits: first(bitset.FromString("1111111100000000"))},
		Solution{Fitness: 10, Bits: first(bitset.FromString("0000111100001111"))},
	},
}

var deceptiveFOS = [][]int{
	[]int{0, 1, 2, 3},
	[]int{4, 5, 6, 7},
	[]int{8, 9, 10, 11},
	[]int{12, 13, 14, 15},
}

func TestDeceptiveLinkage(t *testing.T) {
	freqs := Frequencies(&deceptivePopulation)
	lt := LinkageTree(&deceptivePopulation, freqs)

	for i := 0; i < len(lt); i++ {
		sort.Ints(lt[i])
	}

	for _, dt := range deceptiveFOS {
		found := false

		for _, subset := range lt {
			if IntArrayEquals(subset, dt) {
				found = true
			}
		}

		if !found {
			t.Logf("Linkage Tree:\n %v", lt)
			t.Errorf("Deceptive trap not detected.")
		}
	}
}

// TestHierarchicalStructure ensures that the pure linkage tree created by the
// Linkage Tree function has the properties of an hierarchical clustering.
func TestHierarchicalStructure(t *testing.T) {

	for i := 1; i < 32; i++ {
		pop := NewPopulation(32, i)
		freqs := Frequencies(pop)
		lt := LinkageTree(pop, freqs)

		expected := 2*i - 1
		if len(lt) != expected {
			t.Errorf("len(FOS) = %d, expected %d", len(lt), expected)
		}

		numSingletons := 0
		for _, subset := range lt {
			if len(subset) > 1 {
				break
			}
			numSingletons++
		}

		fmt.Sprintf("")

		if numSingletons != i {
			t.Errorf("FOS contained %d singleton subsets, expected %d", numSingletons, i)
		}

		merged := make([]bool, expected)

		for j := numSingletons; j < expected; j++ {

			parent := append([]int(nil), lt[j]...)
			sort.Ints(parent)

			match := false

			// Search the unmerged clusters for the parent's two children.
			// Test will fail if the FOS ordering was changed.
			for k := j - 1; k >= 0; k-- {

				if merged[k] {
					continue
				}

				right := lt[k]

				for l := k - 1; l >= 0; l-- {

					if merged[l] {
						continue
					}

					left := lt[l]

					if len(left)+len(right) != len(parent) {
						continue
					}

					reconstruct := append([]int(nil), left...)
					reconstruct = append(reconstruct, right...)
					sort.Ints(reconstruct)

					if IntArrayEquals(reconstruct, parent) {
						match = true
						merged[l], merged[k] = true, true
						break
					}
				}

				if match {
					break
				}
			}

			if !match {
				t.Errorf("Subset was not a result of merging two unmerged nodes. %v", parent)
			}
		}

		sum := 0

		for _, v := range lt {
			sum += len(v)
		}

		boundSum := (i*i + 3*i - 2) / 2
		if sum > boundSum {
			t.Errorf("Sum of problem variables is %d, exceeds bound %d.", sum, boundSum)
		}
	}
}
