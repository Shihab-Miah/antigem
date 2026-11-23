package localai

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"
)

// RunnerConfig configures the on-device llama-server process.
type RunnerConfig struct {
	ModelPath  string `json:"model_path"`
	Host       string `json:"host"`
	Port       int    `json:"port"`
	Threads    int    `json:"threads"`
	CtxSize    int    `json:"ctx_size"`
	Alias      string `json:"alias"`
	CacheTypeK string `json:"cache_type_k"`
	CacheTypeV string `json:"cache_type_v"`
}

// RunnerState holds the current runtime state of the local inference server.
type RunnerState struct {
	Running    bool          `json:"running"`
	PID        int           `json:"pid,omitempty"`
	Config     *RunnerConfig `json:"config,omitempty"`
	StartedAt  *time.Time    `json:"started_at,omitempty"`
	Endpoint   string        `json:"endpoint,omitempty"`
	LastError  string        `json:"last_error,omitempty"`
}

// Runner manages the lifecycle of the on-device llama-server binary.
type Runner struct {
	mu     sync.Mutex
	cmd    *exec.Cmd
	cfg    *RunnerConfig
	start  time.Time
	lastErr string
}

// NewRunner creates a new instance of Runner.
func NewRunner() *Runner {
	return &Runner{}
}

// State returns the current execution state of the runner.
func (r *Runner) State() RunnerState {
	r.mu.Lock()
	defer r.mu.Unlock()

	running := r.isRunningLocked()
	st := RunnerState{
		Running:   running,
		Config:    r.cfg,
		LastError: r.lastErr,
	}
	if running && r.cmd != nil && r.cmd.Process != nil {
		st.PID = r.cmd.Process.Pid
		t := r.start
		st.StartedAt = &t
		if r.cfg != nil {
			st.Endpoint = fmt.Sprintf("http://%s:%d/v1", r.cfg.Host, r.cfg.Port)
		}
	}
	return st
}

func (r *Runner) isRunningLocked() bool {
	if r.cmd == nil || r.cmd.Process == nil {
		return false
	}
	// Check if process is still alive without blocking
	if err := r.cmd.Process.Signal(os.Signal(nil)); err != nil {
		return false
	}
	return true
}

// Start launches llama-server with the specified configuration.
func (r *Runner) Start(cfg RunnerConfig) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.isRunningLocked() {
		return fmt.Errorf("local inference server is already running with PID %d", r.cmd.Process.Pid)
	}

	if _, err := os.Stat(cfg.ModelPath); os.IsNotExist(err) {
		return fmt.Errorf("model file does not exist at: %s", cfg.ModelPath)
	}

	if cfg.Host == "" {
		cfg.Host = "127.0.0.1"
	}
	if cfg.Port <= 0 {
		cfg.Port = 8080
	}
	if cfg.Threads <= 0 {
		cfg.Threads = 4
	}
	if cfg.CtxSize <= 0 {
		cfg.CtxSize = 4096
	}
	if cfg.Alias == "" {
		cfg.Alias = "local-model"
	}
	if cfg.CacheTypeK == "" {
		cfg.CacheTypeK = "q8_0"
	}
	if cfg.CacheTypeV == "" {
		cfg.CacheTypeV = "q8_0"
	}

	args := []string{
		"-m", cfg.ModelPath,
		"--host", cfg.Host,
		"--port", fmt.Sprintf("%d", cfg.Port),
		"-c", fmt.Sprintf("%d", cfg.CtxSize),
		"-t", fmt.Sprintf("%d", cfg.Threads),
		"-np", "1",
		"--alias", cfg.Alias,
		"-ctk", cfg.CacheTypeK,
		"-ctv", cfg.CacheTypeV,
	}

	cmd := exec.Command("llama-server", args...)
	cmd.SysProcAttr = nil

	if err := cmd.Start(); err != nil {
		r.lastErr = err.Error()
		return fmt.Errorf("failed to start llama-server: %w", err)
	}

	r.cmd = cmd
	r.cfg = &cfg
	r.start = time.Now()
	r.lastErr = ""

	// Background monitor to clean up once process exits
	go func() {
		_ = cmd.Wait()
		r.mu.Lock()
		if r.cmd == cmd {
			r.cmd = nil
		}
		r.mu.Unlock()
	}()

	return nil
}

// Stop gracefully terminates the running llama-server process.
func (r *Runner) Stop() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.isRunningLocked() {
		return nil
	}

	proc := r.cmd.Process
	if err := proc.Signal(os.Interrupt); err != nil {
		_ = proc.Kill()
	}

	// Wait with timeout
	done := make(chan error, 1)
	go func() {
		_, err := proc.Wait()
		done <- err
	}()

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		_ = proc.Kill()
	}

	r.cmd = nil
	return nil
}

// ProbeReady checks if the local server is answering health probes.
func (r *Runner) ProbeReady(ctx context.Context) bool {
	st := r.State()
	if !st.Running || st.Config == nil {
		return false
	}
	url := fmt.Sprintf("http://%s:%d/v1/models", st.Config.Host, st.Config.Port)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}
