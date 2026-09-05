package codec

import (
	"bytes"
	"math"
	"testing"
)

func sine(f Format) []byte {
	pcm := make([]byte, f.PCMBytes())
	for i := 0; i < len(pcm)/2; i++ {
		v := int16(8000 * math.Sin(2*math.Pi*440*float64(i/f.Channels)/float64(f.Rate)))
		pcm[i*2], pcm[i*2+1] = byte(v), byte(v>>8)
	}
	return pcm
}

func mustNew(t *testing.T, name string, rate, channels, nbyte int) Codec {
	t.Helper()
	c, err := New(name, rate, channels, nbyte)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestTypeIDs(t *testing.T) {
	if PCM.ID() != 0 || LC3.ID() != 1 || Opus.ID() != 2 || TypeOf(1) != LC3 || TypeOf(9) != PCM {
		t.Fatal("id mapping wrong")
	}
}

func TestFormats(t *testing.T) {
	cases := []struct {
		name          string
		rate, ch, nby int
		want          Format
	}{
		{"pcm", 16000, 1, 0, Format{PCM, 16000, 1, 20, 640}},
		{"pcm", 48000, 2, 0, Format{PCM, 48000, 2, 20, 3840}},
		{"lc3", 16000, 1, 0, Format{LC3, 16000, 1, 10, 40}},
		{"lc3", 15000, 1, 0, Format{LC3, 16000, 1, 10, 40}}, // snaps
		{"lc3", 48000, 2, MusicNByte, Format{LC3, 48000, 2, 10, 80}},
		{"opus", 44100, 1, 0, Format{Opus, 48000, 1, 20, 80}}, // snaps
		{"opus", 12000, 1, 0, Format{Opus, 12000, 1, 20, 80}},
	}
	for _, c := range cases {
		got := mustNew(t, c.name, c.rate, c.ch, c.nby).Format()

		if got != c.want {
			t.Errorf("New(%s %d %d %d) = %+v, want %+v", c.name, c.rate, c.ch, c.nby, got, c.want)
		}
	}
}

func TestFormatStringAndKbps(t *testing.T) {
	lc3 := mustNew(t, "lc3", 48000, 2, MusicNByte).Format()

	if s := lc3.String(); s != "lc3 48000Hz 2ch 10ms 80B/ch 64kbps/ch" || lc3.Kbps() != 64 {
		t.Fatalf("got %q kbps=%d", s, lc3.Kbps())
	}
	if s := mustNew(t, "pcm", 24000, 1, 0).Format().String(); s != "pcm 24000Hz 1ch 16-bit 20ms" {
		t.Fatalf("got %q", s)
	}
}

func TestUnknownCodec(t *testing.T) {
	if _, err := New("flac", 16000, 1, 0); err == nil {
		t.Fatal("expected error")
	}
}

func TestPCMIsIdentity(t *testing.T) {
	c := mustNew(t, "pcm", 16000, 1, 0)
	data := []byte{1, 2, 3, 4}

	enc, _ := c.Encode(data)
	dec, _ := c.Decode(data)

	if !bytes.Equal(enc, data) || !bytes.Equal(dec, data) {
		t.Fatal("pcm must pass through")
	}
}

func TestRoundTrips(t *testing.T) {
	for _, c := range []Codec{
		mustNew(t, "lc3", 16000, 1, 0),
		mustNew(t, "lc3", 48000, 2, MusicNByte),
		mustNew(t, "opus", 16000, 1, 0),
		mustNew(t, "opus", 48000, 2, MusicNByte),
	} {
		f := c.Format()
		var dec []byte
		for range 5 { // past the codec's algorithmic delay
			enc, err := c.Encode(sine(f))
			if err != nil {
				t.Fatal(err)
			}
			if f.Type == LC3 && len(enc) != f.FrameBytes*f.Channels {
				t.Fatalf("%s: wire frame %d bytes", f, len(enc))
			}
			if dec, err = c.Decode(enc); err != nil {
				t.Fatal(err)
			}
		}

		if len(dec) != f.PCMBytes() {
			t.Fatalf("%s: decoded %d bytes, want %d", f, len(dec), f.PCMBytes())
		}
		var energy float64
		for i := 0; i < len(dec); i += 2 {
			v := float64(int16(uint16(dec[i]) | uint16(dec[i+1])<<8))
			energy += v * v
		}
		if energy == 0 {
			t.Fatalf("%s: decoded audio is silent", f)
		}
	}
}

func TestOpusMusicModeBitrate(t *testing.T) {
	music := mustNew(t, "opus", 48000, 2, MusicNByte).(*opusCodec)
	voice := mustNew(t, "opus", 24000, 1, 0).(*opusCodec)

	if music.bitrate != 128000 || music.Format().FrameBytes != 160 || music.Format().Kbps() != 64 || voice.bitrate != 0 {
		t.Fatalf("music=%+v voice=%+v", music.Format(), voice.Format())
	}
}

func TestLC3SampleRoundingMatchesLiblc3(t *testing.T) {
	if sample(0) != 0 || sample(1) != 32767 || sample(-1) != -32768 || sample(0.5) != 16384 || sample(-1.5/32768) != -2 {
		t.Fatal("rounding or clipping wrong")
	}
	c := mustNew(t, "lc3", 16000, 1, 0)
	silence, _ := c.Encode(make([]byte, c.Format().PCMBytes()))
	if empty, err := c.Decode(nil); err != nil || len(empty) != c.Format().PCMBytes() || len(silence) != 40 {
		t.Fatalf("empty frame: %d bytes, %v; silence packet %d bytes", len(empty), err, len(silence))
	}
}

func TestEncodeShortFrame(t *testing.T) {
	if _, err := mustNew(t, "lc3", 16000, 1, 0).Encode(make([]byte, 10)); err == nil {
		t.Fatal("expected error")
	}
	if _, err := mustNew(t, "opus", 16000, 1, 0).Encode(make([]byte, 10)); err == nil {
		t.Fatal("expected error")
	}
}
