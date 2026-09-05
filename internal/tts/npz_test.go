package tts

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"testing"
)

func npyBytes(shape []int, data []float32) []byte {
	shapeStr := ""
	for i, d := range shape {
		if i > 0 {
			shapeStr += ", "
		}
		shapeStr += fmt.Sprint(d)
	}
	if len(shape) == 1 {
		shapeStr += ","
	}
	header := fmt.Sprintf("{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }", shapeStr)
	for (10+len(header)+1)%64 != 0 {
		header += " "
	}
	header += "\n"
	var b bytes.Buffer
	b.WriteString("\x93NUMPY")
	b.WriteByte(1)
	b.WriteByte(0)
	binary.Write(&b, binary.LittleEndian, uint16(len(header)))
	b.WriteString(header)
	for _, f := range data {
		binary.Write(&b, binary.LittleEndian, math.Float32bits(f))
	}
	return b.Bytes()
}

func TestParseNPY(t *testing.T) {
	arr, err := ParseNPY(npyBytes([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6}))

	if err != nil || len(arr.Shape) != 2 || arr.Shape[1] != 3 || arr.Data[5] != 6 {
		t.Fatalf("got %+v, %v", arr, err)
	}
}

func TestParseNPYRejectsGarbage(t *testing.T) {
	if _, err := ParseNPY([]byte("nope")); err == nil {
		t.Fatal("expected error")
	}
}

func TestParseNPZ(t *testing.T) {
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, _ := zw.Create("af_heart.npy")
	w.Write(npyBytes([]int{2, 1, 2}, []float32{0.1, 0.2, 0.3, 0.4}))
	zw.Close()

	voices, err := ParseNPZ(buf.Bytes())

	if err != nil || voices["af_heart"] == nil || voices["af_heart"].Shape[0] != 2 || voices["af_heart"].Data[3] != 0.4 {
		t.Fatalf("got %+v, %v", voices, err)
	}
}

func TestFloat16To32(t *testing.T) {
	if float16to32(0x3C00) != 1 || float16to32(0xC000) != -2 || float16to32(0) != 0 {
		t.Fatal("float16 conversion wrong")
	}
}
