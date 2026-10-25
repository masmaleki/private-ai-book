# One command per step. All data is synthetic example data.
PY ?= python3

all: setup train eval test

setup:   ; $(PY) -m refproj.cli setup
train:   ; $(PY) -m refproj.cli train
eval:    ; $(PY) -m refproj.cli eval
demo:    ; $(PY) -m refproj.cli demo
test:    ; $(PY) -m pytest -q
serve:   ; $(PY) -m uvicorn refproj.api:app --host 127.0.0.1 --port 8100
mcp:     ; SESSION_TOKEN=tok-sales $(PY) -m refproj.mcp_server
clean:   ; rm -rf data artifacts .pytest_cache

.PHONY: all setup train eval demo test serve mcp clean
