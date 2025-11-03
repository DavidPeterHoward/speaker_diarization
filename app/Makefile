# AudioTranscribe Makefile
# Simplifies common development and usage tasks

.PHONY: install install-full test run run-test-server clean help

# Default Python interpreter
PYTHON = python
VENV_DIR = venv
VENV_PYTHON = $(VENV_DIR)/bin/python
VENV_PIP = $(VENV_DIR)/bin/pip

# Windows-specific settings
ifeq ($(OS),Windows_NT)
	VENV_PYTHON = $(VENV_DIR)/Scripts/python
	VENV_PIP = $(VENV_DIR)/Scripts/pip
endif

# Default target
help:
	@echo "AudioTranscribe Makefile"
	@echo "------------------------"
	@echo "Available targets:"
	@echo "  install       - Install with basic dependencies (mock backends)"
	@echo "  install-full  - Install with full dependencies (ML backends)"
	@echo "  install-gpu   - Install with full dependencies and GPU support"
	@echo "  test          - Run dependency tests to check what's available"
	@echo "  run           - Run the web server"
	@echo "  run-test      - Run the test web server with sample data"
	@echo "  clean         - Clean up temporary files and cache"
	@echo "  help          - Show this help message"

# Create virtual environment
$(VENV_DIR):
	$(PYTHON) -m venv $(VENV_DIR)

# Install basic dependencies
install: $(VENV_DIR)
	$(PYTHON) install.py --mode basic --venv $(VENV_DIR)

# Install full dependencies (with ML backends)
install-full: $(VENV_DIR)
	$(PYTHON) install.py --mode full --venv $(VENV_DIR)

# Install full dependencies with GPU support
install-gpu: $(VENV_DIR)
	$(PYTHON) install.py --mode full --gpu --venv $(VENV_DIR)

# Run dependency tests
test: $(VENV_DIR)
	$(VENV_PYTHON) test_dependencies.py

# Run the web server
run: $(VENV_DIR)
	$(VENV_PYTHON) main.py --server

# Run the test web server with sample data
run-test: $(VENV_DIR)
	$(VENV_PYTHON) test_webapp.py

# Clean up temporary files and cache
clean:
	rm -rf data/audio_cache/*
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete