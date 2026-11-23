package localai

import (
	"encoding/json"
	"net/http"
	"path/filepath"
	"strings"
)

// Manager manages local AI model discovery and inference runner endpoints.
type Manager struct {
	HomeDir string
	runner  *Runner
}

// NewManager creates a new Manager instance.
func NewManager(homeDir string) *Manager {
	return &Manager{
		HomeDir: homeDir,
		runner:  NewRunner(),
	}
}

// ServeHTTP routes /api/localai/ requests.
func (m *Manager) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/api/localai")
	path = strings.TrimPrefix(path, "/")

	w.Header().Set("Content-Type", "application/json")

	switch {
	case path == "models" && r.Method == http.MethodGet:
		m.handleListModels(w, r)
	case path == "status" && r.Method == http.MethodGet:
		m.handleStatus(w, r)
	case path == "start" && r.Method == http.MethodPost:
		m.handleStart(w, r)
	case path == "stop" && r.Method == http.MethodPost:
		m.handleStop(w, r)
	default:
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"error": "unknown localai endpoint",
		})
	}
}

func (m *Manager) handleListModels(w http.ResponseWriter, r *http.Request) {
	customDirs := []string{
		filepath.Join(m.HomeDir, "models"),
	}
	models := ScanLocalModels(customDirs...)
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"models":  models,
	})
}

func (m *Manager) handleStatus(w http.ResponseWriter, r *http.Request) {
	st := m.runner.State()
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"state":   st,
	})
}

func (m *Manager) handleStart(w http.ResponseWriter, r *http.Request) {
	var req RunnerConfig
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"success": false,
			"error":   "invalid JSON payload",
		})
		return
	}

	if err := m.runner.Start(req); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"success": false,
			"error":   err.Error(),
		})
		return
	}

	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"state":   m.runner.State(),
	})
}

func (m *Manager) handleStop(w http.ResponseWriter, r *http.Request) {
	if err := m.runner.Stop(); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"success": false,
			"error":   err.Error(),
		})
		return
	}

	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"message": "Local inference server stopped",
	})
}
