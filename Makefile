.PHONY: run build test check fmt lint clean release

# Default target
all: check test build

# Run the application
run:
	cargo run

# Build the application
build:
	cargo build

# Run tests
test:
	cargo test

# Check if the project compiles
check:
	cargo check

# Format code
fmt:
	cargo fmt --all

# Run clippy lints
lint:
	cargo clippy -- -D warnings

# Clean build artifacts
clean:
	cargo clean

# Build release version
release:
	cargo build --release

# Install dependencies (useful for CI)
deps:
	rustup component add clippy rustfmt

# Run all checks (useful for pre-commit)
check-all: fmt lint test 