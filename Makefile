.PHONY: help \
	test-unit test-component test-contract test-integration test-e2e test-frontend-all test-layers-stats \
	test-rust-unit test-rust-integration test-rust-e2e test-rust-layers-stats \
	tdd tdd-file tdd-rust tdd-rust-filter test-layer-budget test-rust-layer-budget test-layer-stats test-all-layers

help:
	@echo "Lime test layers"
	@echo "  make tdd                  Run frontend unit tests for local/AI TDD"
	@echo "  make tdd-file FILE=...    Run one frontend unit test file for local/AI TDD"
	@echo "  make tdd-rust             Run Rust unit tests for local/AI TDD"
	@echo "  make tdd-rust-filter FILTER=... [CRATE=...] Run targeted Rust unit tests"
	@echo "  make test-unit            Run frontend pure unit layer"
	@echo "  make test-component       Run React component layer"
	@echo "  make test-contract        Run frontend contract layer"
	@echo "  make test-integration     Run frontend integration layer"
	@echo "  make test-e2e             Run frontend e2e/live-gated layer"
	@echo "  make test-rust-unit       Run Rust unit layer"
	@echo "  make test-rust-integration Run Rust integration layer"
	@echo "  make test-rust-e2e        Run Rust e2e/live-gated layer"
	@echo "  make test-layer-budget    Guard against new component VM migration candidates"
	@echo "  make test-rust-layer-budget Guard Rust e2e tests from default TDD path"
	@echo "  make test-layer-stats     Print frontend and Rust layer stats"

tdd: test-unit

tdd-file:
	@test -n "$(FILE)" || (echo "FILE is required. Example: make tdd-file FILE=scripts/run-vitest-layer.unit.test.mjs" && exit 1)
	npm run test:unit -- "$(FILE)"

tdd-rust: test-rust-layer-budget test-rust-unit

tdd-rust-filter:
	@test -n "$(FILTER)" || (echo "FILTER is required. Example: make tdd-rust-filter FILTER=workspace_support::tests::sanitize_project_dir_name_should_replace_invalid_chars" && exit 1)
	$(MAKE) test-rust-layer-budget
	@if [ -n "$(CRATE)" ]; then \
		npm run test:rust:unit -- -p "$(CRATE)" "$(FILTER)"; \
	else \
		npm run test:rust:unit -- "$(FILTER)"; \
	fi

test-unit:
	npm run test:unit

test-component:
	npm run test:component

test-contract:
	npm run test:contract

test-integration:
	npm run test:integration

test-e2e:
	npm run test:e2e

test-frontend-all:
	npm run test:frontend:all

test-layers-stats:
	npm run test:layers:stats

test-rust-unit:
	npm run test:rust:unit

test-rust-integration:
	npm run test:rust:integration

test-rust-e2e:
	npm run test:rust:e2e

test-rust-layers-stats:
	npm run test:rust:layers:stats

test-layer-budget:
	node scripts/check-vitest-layer-budget.mjs --max-component-candidates 37

test-rust-layer-budget:
	node scripts/check-rust-layer-budget.mjs --max-e2e-runnable 0

test-layer-stats: test-layers-stats test-rust-layers-stats

test-all-layers: test-layer-budget test-rust-layer-budget test-unit test-component test-contract test-integration test-e2e test-rust-unit test-rust-integration test-rust-e2e
