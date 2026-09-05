// Package music implements music search, queueing and streaming to devices.
package music

// MusicTrack is a single music track.
type MusicTrack struct {
	Title           string
	Artist          string
	Album           string
	DurationSeconds int
	VideoID         string // YouTube video ID for yt-dlp streaming
	SongID          string // Apple Music / Spotify song ID
	Service         string // "youtube", "apple" or "spotify"
}
