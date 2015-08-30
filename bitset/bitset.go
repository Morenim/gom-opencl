// Package bitset provides an interface for bit-string data-structures and
// also provides a memory-efficient bit-string implementation.
package bitset

import (
	"bytes"
	"fmt"
)

const pow uint = 6
const mod uint = 63

type bitSet struct {
	len   int
	array []uint64
}

// BitSet provides an interface for manipulating bit-strings by accessing individual
// bits and recombining partial bit-sets.
type BitSet interface {
	// Has tests whether the bit at pos has been set.
	Has(pos int) bool
	// Set sets the bit at pos to one.
	Set(pos int)
	// Clear sets the bit at pos to zero.
	Clear(pos int)
	// CopyBit copies the bit at index from the source into the bit-string.
	CopyBit(src BitSet, index int)
	// CopyBits copies the bits at indices from the source into the bit-string.
	CopyBits(src BitSet, indices []int)
	// Len returns the length of the bit-string.
	Len() int
}

func (bs *bitSet) Len() int {
	return bs.len
}

func (bs *bitSet) Set(pos int) {
	bs.array[pos>>pow] |= (1 << (uint(pos) & mod))
}

func (bs *bitSet) Clear(pos int) {
	bs.array[pos>>pow] &^= (1 << (uint(pos) & mod))
}

func (bs *bitSet) CopyBit(src BitSet, index int) {
	if src.Has(index) {
		bs.Set(index)
	} else {
		bs.Clear(index)
	}
}

func (bs *bitSet) CopyBits(src BitSet, indices []int) {
	for _, index := range indices {
		bs.CopyBit(src, index)
	}
}

func (bs *bitSet) Has(pos int) bool {
	return (bs.array[pos>>pow]&(1<<(uint(pos)&mod)) != 0)
}

func (bs *bitSet) String() string {
	var buffer bytes.Buffer
	skipFirst := 0
	if bs.len%64 != 0 {
		format := fmt.Sprintf("%c0%db", '%', uint((bs.len)+1)%64-1)
		buffer.WriteString(fmt.Sprintf(format, bs.array[len(bs.array)-1]))
		skipFirst = 1
	}
	for i := len(bs.array) - 1 - skipFirst; i >= 0; i-- {
		buffer.WriteString(fmt.Sprintf("%064b", bs.array[i]))
	}
	return buffer.String()
}

// New returns an interface to the dense bit-string implementation.
func New(len int) BitSet {
	return &bitSet{len, make([]uint64, (len+63)/64)}
}

// FromString converts a string in big-endian notation to a new bit-set.
func FromString(s string) (BitSet, error) {
	b := New(len(s))
	for i, c := range s {
		if c == '1' {
			b.Set(len(s) - 1 - i)
		} else if c != '0' {
			format := "bitset: invalid character %v in string encoding"
			return nil, fmt.Errorf(format, c)
		}
	}
	return b, nil
}

func FromUInt32s(ints []uint32, length int) (BitSet, error) {
	r := make([]uint64, 1+((length-1)>>6))
	for i, _ := range r {
		r[i] = uint64(ints[i<<1])
		if i<<1+1 < len(ints) {
			r[i] += (uint64(ints[i<<1+1]) << 32)
		}
	}
	return &bitSet{length, r}, nil
}
