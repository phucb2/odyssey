# Odyssey — common uv/python targets
# Usage: make <target>

.PHONY: help install sync run compile package colab-nb update test clean

UV ?= uv

help: ## Show available targets
	@echo "Targets:"
	@echo "  install   Create venv and install project + deps"
	@echo "  sync      Sync lockfile and environment"
	@echo "  run       Run the package (python -m odyssey)"
	@echo "  compile   Byte-compile sources under src/"
	@echo "  package   Build sdist and wheel into dist/"
	@echo "  colab-nb  Embed src/odyssey into notebooks/colab_odyssey.py and .ipynb"
	@echo "  update    Refresh ODYSSEY_B64 in existing Colab artifacts"
	@echo "  test      Run pytest"
	@echo "  clean     Remove build artifacts and caches"

install: ## Install project into .venv
	$(UV) sync

sync: ## Sync environment from lockfile
	$(UV) sync

run: ## Run library entrypoint
	$(UV) run python -m odyssey

compile: ## Compile Python sources to bytecode
	$(UV) run python -m compileall -q src

package: ## Build distributable packages (sdist + wheel)
	$(UV) build

colab-nb: ## Embed src/odyssey into notebooks/colab_odyssey.py and .ipynb
	$(UV) run python scripts/build_colab_notebook.py

update: ## Refresh ODYSSEY_B64 in notebooks/colab_odyssey.py and .ipynb
	$(UV) run python scripts/update_colab_payload.py

test: ## Run the test suite
	$(UV) run pytest

clean: ## Remove caches and build outputs
	$(UV) run python -c "import pathlib,shutil; roots=[pathlib.Path('dist'),pathlib.Path('build'),pathlib.Path('.pytest_cache')]; \
[shutil.rmtree(p, ignore_errors=True) for p in roots]; \
[shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; \
[shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('*.egg-info')]"
	@echo Cleaned.
