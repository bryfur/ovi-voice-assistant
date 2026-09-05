package models

import (
	"bufio"
	"io"
)

func newBufferedReader(r io.Reader) *bufio.Reader {
	return bufio.NewReaderSize(r, 1<<20)
}
