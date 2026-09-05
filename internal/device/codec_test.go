package device

import "testing"

func TestParseCodecType(t *testing.T) {
	cases := map[string]CodecType{"pcm": PCM, "lc3": LC3, "opus": Opus}

	for in, want := range cases {
		got, err := parseCodecType(in)

		if err != nil || got != want {
			t.Fatalf("parseCodecType(%q) = %v, %v; want %v", in, got, err, want)
		}
	}
}

func TestParseCodecTypeInvalid(t *testing.T) {
	_, err := parseCodecType("flac")

	if err == nil {
		t.Fatal("expected error for unknown codec")
	}
}

func TestNearestValidRate(t *testing.T) {
	cases := []struct {
		rate  int
		valid []int
		want  int
	}{
		{16000, lc3ValidRates, 16000},
		{15000, lc3ValidRates, 16000},
		{9000, lc3ValidRates, 8000},
		{12000, lc3ValidRates, 8000}, // midpoint picks lower
		{1, lc3ValidRates, 8000},
		{96000, lc3ValidRates, 48000},
		{16000, opusValidRates, 16000},
		{11000, opusValidRates, 12000},
		{44100, opusValidRates, 48000},
	}

	for _, c := range cases {
		got := nearestValidRate(c.rate, c.valid)

		if got != c.want {
			t.Errorf("nearestValidRate(%d) = %d, want %d", c.rate, got, c.want)
		}
	}
}

func TestCreatePCMAnyRate(t *testing.T) {
	c, err := newCodec(PCM, 44100, 1, 0)

	if err != nil {
		t.Fatal(err)
	}
	if _, ok := c.(*PCMCodec); !ok || c.SampleRate() != 44100 {
		t.Fatalf("got %T @ %d", c, c.SampleRate())
	}
}

func TestCreateLC3SnapsRate(t *testing.T) {
	c, err := newCodec(LC3, 15000, 1, 0)

	if err != nil {
		t.Fatal(err)
	}
	if c.Type() != LC3 || c.SampleRate() != 16000 {
		t.Fatalf("got %s @ %d", c.Type(), c.SampleRate())
	}
}

func TestCreateOpusSnapsRate(t *testing.T) {
	c, err := newCodec(Opus, 44100, 1, 0)

	if err != nil {
		t.Fatal(err)
	}
	if c.Type() != Opus || c.SampleRate() != 48000 {
		t.Fatalf("got %s @ %d", c.Type(), c.SampleRate())
	}
}

func TestCreateMusicSettings(t *testing.T) {
	lc3, err := newCodec(LC3, 48000, 2, LC3MusicNByte)
	if err != nil || lc3.EncodedFrameBytes() != 80 || lc3.Channels() != 2 {
		t.Fatalf("lc3 music: %+v, %v", lc3, err)
	}

	op, err := newCodec(Opus, 48000, 2, LC3MusicNByte)

	if err != nil || op.(*opusCodec).Bitrate() != 128000 {
		t.Fatalf("opus music: %v, %v", op, err)
	}
}

func TestDescribe(t *testing.T) {
	lc3, _ := newCodec(LC3, 48000, 2, LC3MusicNByte)
	op, _ := newCodec(Opus, 24000, 1, 0)

	if got := Describe(lc3); got != "lc3 48000Hz 2ch 10ms 80B/ch 64kbps/ch" {
		t.Fatalf("lc3: %q", got)
	}
	if got := Describe(NewPCMCodec(24000, 1)); got != "pcm 24000Hz 1ch 16-bit 20ms" {
		t.Fatalf("pcm: %q", got)
	}
	if got := Describe(op); got != "opus 24000Hz 1ch 20ms 80B/ch 32kbps/ch" {
		t.Fatalf("opus: %q", got)
	}
}

func TestCreateNamedUnknown(t *testing.T) {
	_, err := NewCodec("flac", 16000, 1, 0)

	if err == nil {
		t.Fatal("expected error")
	}
}

func TestNameForID(t *testing.T) {
	if NameForID(0) != PCM || NameForID(1) != LC3 || NameForID(2) != Opus || NameForID(9) != PCM {
		t.Fatal("NameForID mapping wrong")
	}
}
