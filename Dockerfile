# MTG Scanner -- single-process container.
#
#   /app/server   API + static file server (Node, TypeScript run natively)
#   /app/dist     built frontend, served by the same process
#   /data         volume: SQLite, scan photos, and the provisioned index/models
#
# Only port 3000 is exposed. Provision /data first (see docs/phase5-interface.md).

# Stage 1: compile the frontend (installs web/ and its build tooling).
FROM node:24-slim AS build
WORKDIR /app
RUN corepack enable
COPY app/package.json app/pnpm-lock.yaml app/pnpm-workspace.yaml ./
COPY app/web/package.json web/
COPY app/server/package.json server/
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
	pnpm install --frozen-lockfile --filter web
COPY app/ ./
RUN pnpm build

# Stage 2: the server's runtime dependencies, nothing else.
FROM node:24-slim AS prod-deps
WORKDIR /app
RUN corepack enable
COPY app/package.json app/pnpm-lock.yaml app/pnpm-workspace.yaml ./
COPY app/web/package.json web/
COPY app/server/package.json server/
# `pnpm deploy` writes the server project plus only its own dependencies to
# /out (the hoisted workspace install would include web's packages too).
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
	pnpm --filter server deploy --prod --legacy /out
# onnxruntime-node ships prebuilt binaries for every OS/CPU (darwin, win32,
# linux x64 + arm64, ~280 MB) plus GPU execution providers. This image uses
# exactly one platform directory and the CPU provider; drop the rest.
RUN cd /out && ARCH="$(node -p process.arch)" \
	&& for base in $(find node_modules -type d -path '*/onnxruntime-node/bin/napi-v6'); do \
		for os in "$base"/*; do for arch in "$os"/*; do \
			[ "$arch" = "$base/linux/$ARCH" ] || rm -rf "$arch"; \
		done; done; \
	done \
	&& find node_modules -name 'libonnxruntime_providers_cuda.so' -delete \
	&& find node_modules -name 'libonnxruntime_providers_tensorrt.so' -delete

# Stage 3: runtime.
FROM node:24-slim AS runtime
LABEL org.opencontainers.image.title="MTG Scanner"
WORKDIR /app
COPY app/package.json ./
COPY --from=prod-deps /out/node_modules ./node_modules
COPY app/server ./server
COPY app/shared ./shared
COPY --from=build /app/dist ./dist
ENV NODE_ENV=production \
	DATA_DIR=/data \
	WEB_DIST=/app/dist \
	HOST=0.0.0.0 \
	PORT=3000
VOLUME ["/data"]
EXPOSE 3000
USER node
CMD ["node", "server/src/index.ts"]
