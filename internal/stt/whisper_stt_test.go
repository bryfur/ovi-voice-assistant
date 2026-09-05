package stt

import (
	"context"
	"encoding/binary"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

type fakeTranscriber struct {
	wav   []byte
	model string
	text  string
}

func (f *fakeTranscriber) Transcribe(_ context.Context, wav []byte, model, language string) (string, error) {
	f.wav = wav
	f.model = model
	return f.text, nil
}

func TestWAVHeader(t *testing.T) {
	pcm := []byte{1, 2, 3, 4}

	wav := WAV(pcm, 16000, 1)

	if string(wav[:4]) != "RIFF" || string(wav[8:12]) != "WAVE" || len(wav) != 44+4 {
		t.Fatalf("bad header: %v", wav[:12])
	}
	if binary.LittleEndian.Uint32(wav[24:]) != 16000 || binary.LittleEndian.Uint16(wav[22:]) != 1 || binary.LittleEndian.Uint32(wav[40:]) != 4 {
		t.Fatal("header fields wrong")
	}
}

func TestWhisperTranscribeShortAudioIsEmpty(t *testing.T) {
	w := NewWhisperSTT(config.Default())
	w.SetClient(&fakeTranscriber{text: "x"})

	text, err := w.Transcribe(make([]byte, 100))

	if err != nil || text != "" {
		t.Fatalf("got %q, %v", text, err)
	}
}

func TestWhisperTranscribeStream(t *testing.T) {
	s := config.Default()
	s.STT.Model = "whisper-1"
	w := NewWhisperSTT(s)
	w.LoadVAD = func() (Prober, error) { return fakeProber{}, nil }
	ft := &fakeTranscriber{text: "hello world"}
	w.SetClient(ft)
	if err := w.Load(); err != nil {
		t.Fatal(err)
	}
	w.params.SilenceTimeout = 10 * time.Millisecond
	w.params.MinSpeech = time.Millisecond
	chunks := make(chan []byte, 64)
	chunk := VADChunkSamples * 2
	for i := 0; i < 20; i++ {
		chunks <- speech(chunk)
	}
	started := 0
	go func() {
		time.Sleep(50 * time.Millisecond)
		for i := 0; i < 5; i++ {
			chunks <- silence(chunk)
			time.Sleep(20 * time.Millisecond)
		}
		close(chunks)
	}()

	text, err := w.TranscribeStream(context.Background(), chunks, func() { started++ })

	if err != nil || text != "hello world" || started != 1 || ft.model != "whisper-1" {
		t.Fatalf("text=%q err=%v started=%d model=%q", text, err, started, ft.model)
	}
	if string(ft.wav[:4]) != "RIFF" {
		t.Fatal("expected WAV upload")
	}
}

func TestWhisperTranscribeStreamNoSpeech(t *testing.T) {
	w := NewWhisperSTT(config.Default())
	w.LoadVAD = func() (Prober, error) { return fakeProber{}, nil }
	w.SetClient(&fakeTranscriber{text: "should not be called"})
	w.Load()
	chunks := make(chan []byte)
	close(chunks)

	text, err := w.TranscribeStream(context.Background(), chunks, nil)

	if err != nil || text != "" {
		t.Fatalf("got %q, %v", text, err)
	}
}

func TestWhisperTranscribeStreamCancelled(t *testing.T) {
	w := NewWhisperSTT(config.Default())
	w.LoadVAD = func() (Prober, error) { return fakeProber{}, nil }
	w.SetClient(&fakeTranscriber{})
	w.Load()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := w.TranscribeStream(ctx, make(chan []byte), nil)

	if err == nil {
		t.Fatal("expected cancellation error")
	}
}

func TestCreateProviders(t *testing.T) {
	s := config.Default()
	s.STT.Provider = "whisper"
	if _, err := Create(s); err != nil {
		t.Fatal(err)
	}
	s.STT.Provider = "nemotron"
	if _, err := Create(s); err != nil {
		t.Fatal(err)
	}
	s.STT.Provider = "bogus"

	_, err := Create(s)

	if err == nil {
		t.Fatal("expected error")
	}
}
