package bitset

import (
	"fmt"
	"testing"
)

var parsingTests = []string{
	"11111", "00000", "10101", "10000", "00001",
}

func TestParsing(t *testing.T) {
	for _, s := range parsingTests {
		bs, err := FromString(s)
		testParseError(t, s, err)
		if actual := fmt.Sprint(bs); actual != s {
			t.Logf("%b %b %b %b", 1, 3, 8, 234235)
			t.Errorf("FromString(%q).String() = %q, expected %q.",
				s, actual, s)
		}
	}
}

var modificationTests = []struct {
	input    string
	index    int
	expected string
}{
	{"11111", -3, "10111"},
	{"11001", 2, "11101"},
}

func testParseError(t *testing.T, s string, err error) {
	if err != nil {
		t.Errorf("FromString(%q) returned error %q.", s, err)
	}
}

func TestModification(t *testing.T) {
	for _, test := range modificationTests {
		bs, err := FromString(test.input)

		testParseError(t, test.input, err)

		if test.index < 0 {
			bs.Clear(-test.index)
		} else {
			bs.Set(test.index)
		}
		if actual := fmt.Sprint(bs); actual != test.expected {
			t.Errorf("Set/Clear(%q, %d) = %q, expected %q.",
				test.input, test.index, actual, test.expected)
		}
	}
}

var hasTests = []struct {
	input    string
	index    int
	expected bool
}{
	{"11111", 0, true},
	{"11001", 2, false},
}

func TestSelection(t *testing.T) {
	for _, test := range hasTests {
		bs, err := FromString(test.input)

		testParseError(t, test.input, err)

		actual := bs.Has(test.index)
		if actual != test.expected {
			t.Errorf("Has(%q, %d) = %q, expected %q.",
				test.input, test.index, actual, test.expected)
		}
	}
}

var copyTests = []struct {
	src      string
	dest     string
	indices  []int
	expected string
}{
	{"11111", "00000", []int{0, 2, 4}, "10101"},
	{"11001", "01111", []int{}, "01111"},
	{"11000", "10101", []int{0, 1, 4}, "10100"},
	{"11111", "11001", []int{0, 1, 2, 3, 4}, "11111"},
}

func TestCopy(t *testing.T) {
	for _, test := range copyTests {
		src, err := FromString(test.src)
		testParseError(t, test.src, err)

		dest, err := FromString(test.dest)
		testParseError(t, test.dest, err)

		dest.CopyBits(src, test.indices)
		if actual := fmt.Sprint(dest); actual != test.expected {
			t.Errorf("CopyBits(%q %q %v) = %q, expected %q.",
				test.src, test.dest, test.indices, actual, test.expected)
		}
	}
}
