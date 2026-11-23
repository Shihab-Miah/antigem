package localai

import (
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ModelInfo describes a local GGUF model detected on the device storage.
type ModelInfo struct {
	Name         string    `json:"name"`
	Path         string    `json:"path"`
	SizeBytes    int64     `json:"size_bytes"`
	SizeFormatted string   `json:"size_formatted"`
	ModifiedAt   time.Time `json:"modified_at"`
	Quantization string    `json:"quantization,omitempty"`
}

// DefaultModelSearchPaths lists directories searched for GGUF model files on Android.
var DefaultModelSearchPaths = []string{
	"/storage/emulated/0/Download/Ai Files",
	"/storage/emulated/0/Download",
	"/sdcard/Download/Ai Files",
	"/sdcard/Download",
}

// formatBytes returns a human-readable file size string.
func formatBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return "1 KB"
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	units := []string{"KB", "MB", "GB", "TB"}
	val := float64(b) / float64(div)
	if exp < len(units) {
		return strings.TrimRight(strings.TrimRight(strings.Sprintf("%.2f", val), "0"), ".") + " " + units[exp]
	}
	return "huge"
}

// parseQuantization extracts the quant tag from standard GGUF filename (e.g. Q4_K_M, Q8_0).
func parseQuantization(filename string) string {
	upper := strings.ToUpper(filename)
	quants := []string{"Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q6_K", "FP16", "BF16"}
	for _, q := range quants {
		if strings.Contains(upper, q) {
			return q
		}
	}
	return ""
}

// ScanLocalModels searches common Android storage locations for .gguf model files.
func ScanLocalModels(customPaths ...string) []ModelInfo {
	seen := make(map[string]bool)
	var results []ModelInfo

	paths := append([]string{}, DefaultModelSearchPaths...)
	paths = append(paths, customPaths...)

	for _, dir := range paths {
		if _, err := os.Stat(dir); os.IsNotExist(err) {
			continue
		}

		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}

		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			name := e.Name()
			if strings.HasSuffix(strings.ToLower(name), ".gguf") {
				fullPath := filepath.Join(dir, name)
				if seen[fullPath] {
					continue
				}
				seen[fullPath] = true

				info, err := e.Info()
				if err != nil {
					continue
				}

				results = append(results, ModelInfo{
					Name:          strings.TrimSuffix(name, filepath.Ext(name)),
					Path:          fullPath,
					SizeBytes:     info.Size(),
					SizeFormatted: formatBytes(info.Size()),
					ModifiedAt:    info.ModTime(),
					Quantization:  parseQuantization(name),
				})
			}
		}
	}

	return results
}
