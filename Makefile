# MTG Scanner -- local workflows. The JavaScript app (browser + server) lives
# in app/ as one pnpm package; training/ is the Python side.
#
#   make install        pnpm install in app/
#   make dev            Vite (5173, hot reload) + API server (3000) together
#   make build          Compile the frontend into app/dist
#   make start          Build, then serve app + API from one Node process on :3000
#   make check          Typecheck + lint + unit tests
#   make parity         Compare Node embeddings with the Python-built index
#   make docker-build   Build the container image
#   make docker-run     Run the image with DATA_DIR mounted at /data
#   make bundle         Pack DATA_DIR's index + models into bundle/mtg-scanner-data-<date>.tar.gz
#   make unbundle       Unpack a bundle into DATA_DIR: make unbundle BUNDLE=path/to/file.tar.gz
#
# DATA_DIR must contain index/card_index.faiss, index/card_metadata.parquet,
# models/siglip2-base.onnx and models/card-detector.onnx (see docs/phase5-interface.md).

DATA_DIR ?= $(CURDIR)/data
# Commands run inside app/, so make relative paths absolute first.
override DATA_DIR := $(abspath $(DATA_DIR))
PORT ?= 3000
IMAGE_TAG ?= mtg-scanner:local
APP := app

REQUIRED := index/card_index.faiss index/card_metadata.parquet models/siglip2-base.onnx models/card-detector.onnx

BUNDLE_DIR := bundle

.PHONY: help install dev dev-server dev-web build start test typecheck lint check parity check-data docker-build docker-run compose-up compose-down bundle unbundle

help:
	@sed -n '2,15p' Makefile | sed 's/^# \{0,1\}//'

install:
	cd $(APP) && pnpm install

dev:
	cd $(APP) && DATA_DIR="$(DATA_DIR)" PORT=$(PORT) pnpm dev

dev-server:
	cd $(APP) && DATA_DIR="$(DATA_DIR)" PORT=$(PORT) pnpm dev:server

dev-web:
	cd $(APP) && pnpm dev:web

build:
	cd $(APP) && pnpm build

start: build
	cd $(APP) && DATA_DIR="$(DATA_DIR)" PORT=$(PORT) pnpm start

test:
	cd $(APP) && pnpm test

typecheck:
	cd $(APP) && pnpm typecheck

lint:
	cd $(APP) && pnpm lint

check: typecheck lint test

parity:
	cd $(APP) && DATA_DIR="$(DATA_DIR)" pnpm parity

check-data:
	@for f in $(REQUIRED); do \
		test -f "$(DATA_DIR)/$$f" || { echo "missing: $(DATA_DIR)/$$f"; exit 1; }; \
	done; echo "$(DATA_DIR) has all required files."

docker-build:
	docker build -t $(IMAGE_TAG) .

docker-run: check-data
	docker run --rm -it -p $(PORT):3000 -v "$(DATA_DIR)":/data --name mtg-scanner $(IMAGE_TAG)

compose-up: check-data
	DATA_DIR="$(DATA_DIR)" PORT=$(PORT) docker compose up --build

compose-down:
	docker compose down

# One .tar.gz of the four provisioned files, so another machine can skip the
# training pipeline. About 540 MB; too large for git, share it as a release
# asset or a plain download.
bundle: check-data
	@mkdir -p $(BUNDLE_DIR)
	@name="mtg-scanner-data-$$(date -u +%Y%m%d)"; \
	tar -C "$(DATA_DIR)" -czf "$(BUNDLE_DIR)/$$name.tar.gz" $(REQUIRED); \
	ls -lh "$(BUNDLE_DIR)/$$name.tar.gz"

unbundle:
	@test -n "$(BUNDLE)" || { echo "usage: make unbundle BUNDLE=path/to/mtg-scanner-data-*.tar.gz [DATA_DIR=...]"; exit 1; }
	@mkdir -p "$(DATA_DIR)"
	tar -C "$(DATA_DIR)" -xzf "$(BUNDLE)"
	@$(MAKE) --no-print-directory check-data
