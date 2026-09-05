package codec

// pcm passes 16-bit audio through unchanged in 20 ms frames.
type pcm struct{ f Format }

func (p pcm) Format() Format {
	p.f.FrameBytes = p.f.PCMBytes()
	return p.f
}

func (pcm) Encode(frame []byte) ([]byte, error) { return frame, nil }
func (pcm) Decode(frame []byte) ([]byte, error) { return frame, nil }
