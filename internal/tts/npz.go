package tts

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// NPYArray is a parsed float32 numpy array.
type NPYArray struct {
	Shape []int
	Data  []float32
}

var npyShapeRe = regexp.MustCompile(`'shape':\s*\(([^)]*)\)`)
var npyDescrRe = regexp.MustCompile(`'descr':\s*'([^']+)'`)
var npyOrderRe = regexp.MustCompile(`'fortran_order':\s*(True|False)`)

// ParseNPY parses a .npy file containing float32 (or float16/float64) data.
func ParseNPY(data []byte) (*NPYArray, error) {
	if len(data) < 10 || string(data[:6]) != "\x93NUMPY" {
		return nil, fmt.Errorf("not an NPY file")
	}
	major := data[6]
	var headerLen int
	var off int
	if major == 1 {
		headerLen = int(binary.LittleEndian.Uint16(data[8:10]))
		off = 10
	} else {
		headerLen = int(binary.LittleEndian.Uint32(data[8:12]))
		off = 12
	}
	if len(data) < off+headerLen {
		return nil, fmt.Errorf("truncated NPY header")
	}
	header := string(data[off : off+headerLen])
	body := data[off+headerLen:]

	if m := npyOrderRe.FindStringSubmatch(header); m != nil && m[1] == "True" {
		return nil, fmt.Errorf("fortran-ordered NPY not supported")
	}
	descr := ""
	if m := npyDescrRe.FindStringSubmatch(header); m != nil {
		descr = m[1]
	}
	var shape []int
	if m := npyShapeRe.FindStringSubmatch(header); m != nil {
		for _, s := range strings.Split(m[1], ",") {
			s = strings.TrimSpace(s)
			if s == "" {
				continue
			}
			n, err := strconv.Atoi(s)
			if err != nil {
				return nil, fmt.Errorf("bad NPY shape %q", m[1])
			}
			shape = append(shape, n)
		}
	}
	count := 1
	for _, d := range shape {
		count *= d
	}
	arr := &NPYArray{Shape: shape, Data: make([]float32, count)}
	switch descr {
	case "<f4", "=f4", "f4":
		if len(body) < count*4 {
			return nil, fmt.Errorf("truncated NPY data")
		}
		for i := range arr.Data {
			arr.Data[i] = math.Float32frombits(binary.LittleEndian.Uint32(body[i*4:]))
		}
	case "<f8", "=f8", "f8":
		if len(body) < count*8 {
			return nil, fmt.Errorf("truncated NPY data")
		}
		for i := range arr.Data {
			arr.Data[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(body[i*8:])))
		}
	case "<f2", "=f2", "f2":
		if len(body) < count*2 {
			return nil, fmt.Errorf("truncated NPY data")
		}
		for i := range arr.Data {
			arr.Data[i] = float16to32(binary.LittleEndian.Uint16(body[i*2:]))
		}
	default:
		return nil, fmt.Errorf("unsupported NPY dtype %q", descr)
	}
	return arr, nil
}

func float16to32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	var bits uint32
	switch exp {
	case 0:
		if mant == 0 {
			bits = sign << 31
		} else {
			// subnormal
			e := uint32(127 - 15 + 1)
			for mant&0x400 == 0 {
				mant <<= 1
				e--
			}
			mant &= 0x3ff
			bits = sign<<31 | e<<23 | mant<<13
		}
	case 0x1f:
		bits = sign<<31 | 0xff<<23 | mant<<13
	default:
		bits = sign<<31 | (exp+127-15)<<23 | mant<<13
	}
	return math.Float32frombits(bits)
}

// ParseNPZ parses a numpy .npz archive into named arrays.
func ParseNPZ(data []byte) (map[string]*NPYArray, error) {
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, fmt.Errorf("npz: %w", err)
	}
	out := map[string]*NPYArray{}
	for _, f := range zr.File {
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}
		b, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			return nil, err
		}
		arr, err := ParseNPY(b)
		if err != nil {
			return nil, fmt.Errorf("npz entry %s: %w", f.Name, err)
		}
		out[strings.TrimSuffix(f.Name, ".npy")] = arr
	}
	return out, nil
}
