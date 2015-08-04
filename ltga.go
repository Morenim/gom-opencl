package main

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"

	"github.com/Morenim/gom-opencl/bitset"
)

// Solution is a bitstring solution to a maximization optimization problem.
type Solution struct {
	Fitness float64
	Bits    bitset.BitSet
}

func (s Solution) String() string {
	return fmt.Sprintf("%v %v\n", s.Fitness, s.Bits)
}

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

var (
	logLookup float64
)

type matrix struct {
	data [][]float64
}

func (m *matrix) get(i, j int) float64 {
	if i <= j {
		i, j = j, i
	}
	return m.data[i][j]
}

func (m *matrix) set(i, j int, value float64) {
	if i <= j {
		i, j = j, i
	}
	m.data[i][j] = value
}

func newMatrix(size int) *matrix {
	m := new(matrix)
	m.data = make([][]float64, size)
	for i := 0; i < len(m.data); i++ {
		m.data[i] = make([]float64, i+1)
	}
	return m
}

func (m *matrix) String() string {
	var buffer bytes.Buffer
	for i := 0; i < len(m.data); i++ {
		for j := 0; j < len(m.data); j++ {
			buffer.WriteString(fmt.Sprintf("%0.05f ", m.get(i, j)))
		}
		buffer.WriteString("\n")
	}
	return buffer.String()
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

// Count the frequencies of bits for every permutation of the problem
// variables indicated by indices. Used to comptue the distance matrix.
func frequency(pop *Population, indices []int) []int {
	numPerms := 1 << uint(len(indices))
	perms := make([]int, numPerms)

	for i := 0; i < pop.Size(); i++ {
		index := 0

		for j := len(indices) - 1; j >= 0; j-- {
			if pop.Solutions[i].Bits.Has(indices[j]) {
				index += 1 << uint(j)
			}
		}
		perms[index]++
	}

	return perms
}

// Compute the Entropy information measure from an array of problem
// variable frequencies.
func entropy(freqs []int, size int) float64 {
	p := 0.0
	for _, f := range freqs {
		if f > 0 {
			p += -(float64(f) / float64(size)) * (math.Log(float64(f)) - math.Log(float64(size)))
		}
	}
	return p
}

func Frequencies(pop *Population) [][][]int {
	freqs := make([][][]int, pop.Length())

	indices1 := make([]int, 1)
	indices2 := make([]int, 2)
	for i := 0; i < pop.Length(); i++ {
		freqs[i] = make([][]int, i+1)
		for j := 0; j < i; j++ {
			indices2[0], indices2[1] = i, j
			freqs[i][j] = frequency(pop, indices2)
		}

		indices1[0] = i
		freqs[i][i] = frequency(pop, indices1)
	}

	return freqs
}

// Function distanceMatrix computes the mutual information between every
// pair of problem variables. The frequencies of bits in the population
// are used for the probabilites.
func distanceMatrix(pop *Population, frequencies [][][]int) *matrix {
	distances := newMatrix(pop.Length())

	for i := 0; i < pop.Length(); i++ {
		for j := 0; j < i; j++ {
			distances.set(i, j, entropy(frequencies[i][j], pop.Size()))
		}

		distances.set(i, i, entropy(frequencies[i][i], pop.Size()))
	}

	for i := 0; i < pop.Length(); i++ {
		for j := 0; j < i; j++ {
			distances.set(i, j, distances.get(i, i)+distances.get(j, j)-distances.get(i, j))
		}
	}

	return distances
}

// Function neighbour finds the nearest neighbour of index in the
// similarity matrix. The value is maximized, and equality is resolved
// based on the lowest size of the two subsets (in the mpm).
func neighbour(index int, sm *matrix, mpm [][]int) int {
	s := 0
	if s == index {
		s++
	}
	for i := 1; i < len(mpm); i++ {
		if i != index && ((sm.get(index, i) > sm.get(index, s)) || ((sm.get(index, i) == sm.get(index, s)) && (len(mpm[i]) < len(mpm[s])))) {
			s = i
		}
	}
	return s
}

func LinkageTree(pop *Population, frequencies [][][]int) [][]int {

	// Validate Input

	switch pop.Length() {
	case 0:
		return nil
	case 1:
		return [][]int{[]int{0}}
	case 2:
		return [][]int{[]int{0}, []int{1}, []int{0, 1}}
	}
	if pop.Length() == 0 {
		return nil
	}

	if pop.Length() == 1 {
	}

	distances := distanceMatrix(pop, frequencies)

	// Array mpm will store all unmerged subsets, starting from the
	// singleton subsets and ending with the set of all problem variables.
	mpm := make([][]int, pop.Length())
	order := rand.Perm(pop.Length())
	for i := 0; i < len(mpm); i++ {
		mpm[i] = make([]int, 1)
		mpm[i][0] = order[i]
	}

	// Array fos will store all singleton subsets and every subset created
	// by merging subsets during the algorithm.
	fos := make([][]int, pop.Length(), pop.Length()+pop.Length()-1)
	for i := 0; i < len(mpm); i++ {
		fos[i] = mpm[i]
	}

	// Similarites contains the similarity measures between the subsets
	// stored in the mpm array.
	sm := newMatrix(pop.Length())
	for i := 0; i < len(mpm); i++ {
		for j := 0; j < len(mpm); j++ {
			sm.set(i, j, distances.get(mpm[i][0], mpm[j][0]))
		}
		sm.set(i, i, 0.0)
	}

	chain := make([]int, pop.Length()+2)
	end := 0
	done := false

	for !done {
		// Chain is empty, so pick a random subset from mpm as the start.
		if end == 0 {
			chain[end] = rand.Intn(len(mpm))
			end++
		}

		// Chain is too small, so append the closest subset from mpm
		// until the chain is size three.
		for end < 3 {
			chain[end] = neighbour(chain[end-1], sm, mpm)
			end++
		}

		// Keep appending the closest subset until the subset at the end of
		// the chain is closest to the second to last subset of the chain.
		for chain[end-3] != chain[end-1] {
			chain[end] = neighbour(chain[end-1], sm, mpm)

			if (sm.get(chain[end-1], chain[end]) == sm.get(chain[end-1], chain[end-2])) && (chain[end] != chain[end-2]) {
				chain[end] = chain[end-2]
			}
			end++
			if end > pop.Length() {
				break
			}
		}

		// Swap the variables so the chain can be altered more easily.
		r0, r1 := chain[end-2], chain[end-1]
		if r0 > r1 {
			r0, r1 = r1, r0
		}
		end -= 3

		if r1 < len(mpm) {
			// Merge the two subsets at the end of the chain.
			subset := append(mpm[r0], mpm[r1]...)

			fos = append(fos, subset)

			sum := float64(len(mpm[r0]) + len(mpm[r1]))
			mul0, mul1 := float64(len(mpm[r0]))/sum, float64(len(mpm[r1]))/sum

			// Adjust the similarity between the merged subset and the
			// remaining subsets based on the length-weighted similarity.
			for i := 0; i < len(mpm); i++ {
				if i != r0 && i != r1 {
					sm.set(i, r0, mul0*sm.get(i, r0)+mul1*sm.get(i, r1))
				}
			}

			// Subset r0 is replaced by the merged subset.
			mpm[r0] = subset

			// Subset r1 is removed unless it was at the end.
			if r1 < len(mpm)-1 {
				mpm[r1] = mpm[len(mpm)-1]

				for i := 0; i < r1; i++ {
					sm.set(i, r1, sm.get(i, len(mpm)-1))
				}

				for i := r1 + 1; i < len(mpm)-1; i++ {
					sm.set(r1, i, sm.get(i, len(mpm)-1))
				}
			}

			// Fix the chain if needed because the last subset was moved.
			for i := 0; i < end; i++ {
				if chain[i] == len(mpm)-1 {
					chain[i] = r1
					break
				}
			}

			// Shrink the mpm (2 subsets removed, one added).
			mpm = mpm[0 : len(mpm)-1]

			if len(mpm) == 1 {
				done = true
			}
		}
	}

	return fos
}
