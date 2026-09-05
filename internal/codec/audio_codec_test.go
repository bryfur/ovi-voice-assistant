package codec

import "testing"

func TestParseCodecType(t *testing.T) {
	cases := map[string]CodecType{"pcm": PCM, "lc3": LC3, "opus": Opus}

	for in, want := range cases {
		got, err := ParseCodecType(in)

		if err != nil || got != want {
			t.Fatalf("ParseCodecType(%q) = %v, %v; want %v", in, got, err, want)
		}
	}
}

func TestParseCodecTypeInvalid(t *testing.T) {
	_, err := ParseCodecType("flac")

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
		{16000, LC3ValidRates, 16000},
		{15000, LC3ValidRates, 16000},
		{9000, LC3ValidRates, 8000},
		{12000, LC3ValidRates, 8000}, // midpoint picks lower
		{1, LC3ValidRates, 8000},
		{96000, LC3ValidRates, 48000},
		{16000, OpusValidRates, 16000},
		{11000, OpusValidRates, 12000},
		{44100, OpusValidRates, 48000},
	}

	for _, c := range cases {
		got := NearestValidRate(c.rate, c.valid)

		if got != c.want {
			t.Errorf("NearestValidRate(%d) = %d, want %d", c.rate, got, c.want)
		}
	}
}

func TestCreatePCMAnyRate(t *testing.T) {
	c, err := Create(PCM, 44100, 1, 0)

	if err != nil {
		t.Fatal(err)
	}
	if _, ok := c.(*PCMCodec); !ok || c.SampleRate() != 44100 {
		t.Fatalf("got %T @ %d", c, c.SampleRate())
	}
}

func TestCreateLC3SnapsRate(t *testing.T) {
	c, err := Create(LC3, 15000, 1, 0)

	if err != nil {
		t.Fatal(err)
	}
	if c.Type() != LC3 || c.SampleRate() != 16000 {
		t.Fatalf("got %s @ %d", c.Type(), c.SampleRate())
	}
}

func TestCreateOpusSnapsRate(t *testing.T) {
	c, err := Create(Opus, 44100, 1, 0)

	if err != nil {
		t.Fatal(err)
	}
	if c.Type() != Opus || c.SampleRate() != 48000 {
		t.Fatalf("got %s @ %d", c.Type(), c.SampleRate())
	}
}

func TestCreateMusicSettings(t *testing.T) {
	lc3, err := Create(LC3, 48000, 2, LC3MusicNByte)
	if err != nil || lc3.EncodedFrameBytes() != 60 || lc3.Channels() != 2 {
		t.Fatalf("lc3 music: %+v, %v", lc3, err)
	}

	op, err := Create(Opus, 48000, 2, LC3MusicNByte)

	if err != nil || op.(*OpusCodec).Bitrate() != 96000 {
		t.Fatalf("opus music: %v, %v", op, err)
	}
}

func TestCreateNamedUnknown(t *testing.T) {
	_, err := CreateNamed("flac", 16000, 1, 0)

	if err == nil {
		t.Fatal("expected error")
	}
}

func TestNameForID(t *testing.T) {
	if NameForID(0) != PCM || NameForID(1) != LC3 || NameForID(2) != Opus || NameForID(9) != PCM {
		t.Fatal("NameForID mapping wrong")
	}
}
