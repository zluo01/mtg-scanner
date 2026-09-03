# Phase 6: Cloud Deployment

**Status**: Planned

> The Phase 6 code (Lambda crates, `store/aws.rs`, `infra/` CDK stacks) was
> removed from the tree during Phase 5. Phase 5 has since replaced the Rust
> backend and the TanStack/Nitro frontend with a single Node process (Hono
> + SolidJS SPA; see phase5-interface.md). The Rust-Lambda-per-operation
> design below is therefore **historical**. The current plan is one
> function (or one container task) running the same Hono app through
> `hono/aws-lambda`, with `DATA_DIR` contents on S3 and the library on a
> managed database. The sections below are kept as the record of the
> earlier design and its cost model; revise them when this phase starts.

## Objective

Deploy the MTG card scanner to AWS serverless infrastructure. Phase 5 proved
the product works end-to-end in a single Docker container; Phase 6 takes the
same shared Rust handler code and wires it into Lambda + DynamoDB + S3 +
CloudFront + Route 53.

The frontend built in Phase 5 is reused with a different Nitro preset
(`aws-lambda` instead of `node-server`). Server functions are re-pointed from
`fetch('http://127.0.0.1:8080/...')` to `LambdaClient.invoke()` calls. No
frontend component or route changes.

The browser runs card detection (YOLO11n) and rectification (OpenCV.js)
client-side, then sends rectified images to the cloud for identification.

## Components

### Cloud Deployment: Multi-Lambda Architecture

The application uses multiple Lambda functions with distinct responsibilities:

- **Nitro Lambda** (Node.js): SSR + passthrough to backend
- **7 Backend Lambdas** (Rust): one per operation, independent concurrency and memory
- **1 Cron Lambda** (Rust): periodic Scryfall data refresh + index rebuild

Request flow:

```
CloudFront (scan.yourdomain.com)
    |
    +-- /assets/* ------> S3 (React bundle, YOLO ONNX model, static assets)
    |
    +-- /* -------------> REST API Gateway (streaming)
                              |
                              v
                         Nitro Lambda (Node.js 22.x, ~50MB)
                              |
                              +-- SSR (React pages)
                              +-- Server functions (pure passthrough):
                              |     Forward to specific backend Lambda (IAM)
                              |     Return response to browser
                              |
                              v (IAM invoke)
                         Backend Lambdas (Rust, provided.al2023, ARM64)
                              +-- scan-card:       S3 index download (cold start) + embed + FAISS + DynamoDB insert + S3 save (or duplicate info)
                              +-- merge-card:      DynamoDB update + optional S3 image swap
                              +-- get-library:     DynamoDB query
                              +-- update-card:     DynamoDB update
                              +-- delete-card:     DynamoDB + S3 delete
                              +-- search-cards:    S3 card metadata search
                              +-- export-library:  DynamoDB -> CSV
                         Cron Lambda (Rust, EventBridge every 2 weeks)
                              +-- cron-update:     Scryfall refresh -> rebuild index -> S3
```

### Why Separate Backend Lambdas

| Concern | Single Lambda | Lambda per operation |
|---------|--------------|---------------------|
| Memory | 2GB for everything | scan: 2GB, CRUD: 128-256MB |
| Cold start | Every CRUD call loads ML model | CRUD: ~5-10ms, scan: ~1-3s (S3 index download) |
| Concurrency | One limit | Independent per operation |
| Deployment | Change CRUD -> redeploy scan model | Independent deploy per operation |
| Cost | Pay 2GB for getLibrary | Pay 128MB for getLibrary |

### What Runs Where

| Component | Location | Size | Latency |
|-----------|----------|------|---------|
| Camera capture | Browser | -- | -- |
| YOLO11n-OBB | Browser (ONNX Runtime Web) | ~6MB (cached) | ~50-150ms |
| OpenCV rectify | Browser (WASM) | ~8MB (cached) | ~10ms |
| SSR + passthrough | Nitro Lambda | ~50MB | ~50-200ms |
| SigLIP2 embed | scan-card Lambda (ort) | ~356MB (S3 -> /tmp on cold start) | est. ~500ms-2s (Graviton) |
| FAISS search | scan-card Lambda (pure-Rust flat IP) | ~331MB (S3 -> /tmp on cold start) | ~3-8ms |
| S3 image save | scan-card Lambda | -- | ~20ms |
| DynamoDB CRUD | Backend Lambdas | -- | ~5-10ms |

### CDK Stack Structure

Three stacks separated by change cadence and dependency:

| Stack | Resources | Changes When |
|-------|-----------|-------------|
| **Data** | DynamoDB table + GSI, S3 data bucket | Rarely (schema changes) |
| **CDN** | CloudFront, S3 assets bucket, S3 images bucket | Domain/CDN changes |
| **API** | REST API Gateway, Nitro Lambda, 7 Backend Lambdas, 1 Cron Lambda | Code deploys |

### Client-Side Pipeline (Browser ML)

Same as Phase 5 -- the browser pipeline is identical between local and cloud.
The only difference is that the YOLO ONNX model is cached from CloudFront
(`/assets/models/yolo11n_obb.onnx`) rather than served by the local axum
server.

| Step | Technology | Size (cached) | Latency |
|------|-----------|--------------|---------|
| Camera access | MediaDevices API | -- | -- |
| Card detection | ONNX Runtime Web (WebGL/WASM) | ~6MB | ~50-150ms |
| Rectification | OpenCV.js (WASM) | ~8MB | ~10ms |
| Image encode | Canvas toBlob (JPEG) | -- | ~5ms |

## Configuration

### Nitro Lambda

| Property | Value |
|----------|-------|
| Runtime | Node.js 22.x (managed, not container) |
| Architecture | ARM64 (Graviton) |
| Memory | 512 MB |
| Timeout | 30 seconds |
| Handler | Nitro-compiled handler (`aws-lambda` preset) |
| Bundle size | ~10-20MB (tree-shaken by Vite) |

Dependencies: TanStack Start + React (SSR), AWS SDK Lambda client (invoke
backend Lambdas).

IAM: `lambda:InvokeFunction` on all 7 backend Lambda ARNs + CloudWatch Logs.
No DynamoDB, no S3.

**Switching from Phase 5 to Phase 6**: change the Nitro preset in
`vite.config.ts` from `node-server` to `aws-lambda`. Update server function
bodies from `fetch('http://127.0.0.1:8080/...')` to AWS SDK Lambda invoke.
The frontend components, routes, and React hooks are unchanged.

### Backend Lambdas (Rust)

All use `provided.al2023` runtime on ARM64 (Graviton). Built with
`cargo-lambda`. scan-card downloads ONNX model + FAISS index +
card_metadata.parquet from S3 on cold start and caches in `/tmp` (1GB
ephemeral storage).

| Lambda | Runtime | Memory | Timeout | Key Dependencies |
|--------|---------|--------|---------|------------------|
| `scan-card` | provided.al2023 (ARM64) | 2048 MB | 30s | ort, aws-sdk-s3, aws-sdk-dynamodb |
| `merge-card` | provided.al2023 (ARM64) | 128 MB | 10s | aws-sdk-dynamodb, aws-sdk-s3 |
| `get-library` | provided.al2023 (ARM64) | 128 MB | 10s | aws-sdk-dynamodb |
| `update-card` | provided.al2023 (ARM64) | 128 MB | 10s | aws-sdk-dynamodb |
| `delete-card` | provided.al2023 (ARM64) | 128 MB | 10s | aws-sdk-dynamodb, aws-sdk-s3 |
| `search-cards` | provided.al2023 (ARM64) | 256 MB | 10s | aws-sdk-s3 (card_metadata.parquet) |
| `export-library` | provided.al2023 (ARM64) | 256 MB | 30s | aws-sdk-dynamodb |
| `cron-update` | provided.al2023 (ARM64) | 2048 MB | 900s | aws-sdk-s3, reqwest (Scryfall API), ort |

Rust crate stack (cloud only):

| Crate | Purpose |
|-------|---------|
| `aws-sdk-dynamodb` | DynamoDB CRUD (official AWS Rust SDK) |
| `aws-sdk-s3` | S3 photo storage (official AWS Rust SDK) |
| `lambda_runtime` + `aws_lambda_events` | Lambda handler framework |
| `cargo-lambda` | Build + deploy tooling |

Shared with Phase 5:
- `ort` (ONNX Runtime)
- `arrow` + `parquet`
- `reqwest`
- `serde` / `serde_json`
- `image`

### scan-card Lambda

The most complex Lambda. Uses `provided.al2023` like all other Lambdas.
On cold start, downloads ONNX model (~356MB), FAISS index (~331MB), and
card_metadata.parquet (~11MB) from S3 data bucket and caches them in `/tmp`
(1GB ephemeral storage). The ONNX model never changes but is stored in S3
for uniform deployment (no container images).

Cold start: ~3-5s (S3 download ~700MB at ~200-400MB/s within same region).
Warm invocations reuse cached files from `/tmp`.

Inference pipeline (identical to Phase 5 axum handler, different storage):
1. Receive `card_id` (client-generated UUID), image bytes, `foil` flag
2. Decode JPEG image bytes
3. Preprocess to 384x384 (SigLIP2 input format)
4. ONNX Runtime inference (SigLIP2 Base p16-384, 768-dim embedding)
5. FAISS search against 108K card embeddings
6. Determine confidence (CONFIDENT / AMBIGUOUS / NO_MATCH)
7. Build card entry based on confidence
8. Check DynamoDB for duplicate (GSI on `scryfall_id`, post-filter on `foil`).
   Only when scryfall_id is non-null. NO_MATCH skips duplicate check.
9. Always insert new card entry in DynamoDB -> save image to S3 -> return
   result. If S3 fails, roll back DynamoDB record within same invocation.

Write ordering: DynamoDB first, then S3. Eliminates orphaned S3 images.

### CRUD Lambdas

Lightweight DynamoDB operations. Each is a single Rust binary (~5-10MB) with
only the AWS SDK as a dependency. Cold start ~5-10ms, execution ~10-50ms.

### search-cards Lambda

Downloads card_metadata.parquet from S3 on cold start, loads the 108K card
name index into memory (~10-20MB). Performs case-insensitive prefix/substring
matching on card names. Returns matching cards with Scryfall metadata.

### IAM Permissions (Backend Lambdas)

| Lambda | Permissions |
|--------|------------|
| `scan-card` | DynamoDB (PutItem, Query on GSI), S3 data (GetObject on index), S3 images (PutObject on scans) |
| `merge-card` | DynamoDB (UpdateItem), S3 images (PutObject, DeleteObject) |
| `get-library` | DynamoDB (GetItem, Query) |
| `update-card` | DynamoDB (UpdateItem) |
| `delete-card` | DynamoDB (DeleteItem), S3 images (DeleteObject) |
| `search-cards` | S3 data (GetObject on card_metadata.parquet) |
| `export-library` | DynamoDB (Query) |
| `cron-update` | S3 data (PutObject on index + parquet files) |

### DynamoDB Table Design

**Table**: `mtg-scanner-cards`

| Key | Type | Description |
|-----|------|-------------|
| Partition key | `card_id` (String) | UUID |

**Attributes**:

| Field | Type | Description |
|-------|------|-------------|
| `scryfall_id` | String | Scryfall card ID (exact printing) |
| `name` | String | Card name (denormalized for display + search) |
| `set_code` | String | Set code (e.g. "m11") |
| `collector_number` | String | Collector number |
| `foil` | Boolean | Foil status (foil + non-foil = separate entries) |
| `count` | Number | Copy count (default: 1) |
| `created_at` | String | ISO timestamp |
| `updated_at` | String | ISO timestamp |

No stored URLs. Image URLs constructed on the fly:
- Scryfall: `https://cards.scryfall.io/normal/front/{id[0]}/{id[1]}/{scryfall_id}.jpg`
- User photo: `https://{cloudfront}/scans/{card_id}.jpg`

**GSI**: `scryfall-index`

| Key | Type | Purpose |
|-----|------|--------|
| Partition key | `scryfall_id` (String) | Duplicate detection during scan |

Foil status checked as post-filter on GSI results.
Billing: On-demand (pay-per-request). Free tier covers 25 RCU + 25 WCU + 25GB.

### S3 Structure

Three S3 buckets separated by access pattern:

**Assets bucket** (public via CloudFront `/assets/*`):
- `index.html`, `assets/` (hashed React bundles)
- `models/yolo11n_obb.onnx` (~6MB, browser detection model)

**Data bucket** (private, backend Lambda access only):
- `index/card_index.faiss` (~331MB, FAISS vector index)
- `index/card_metadata.parquet` (~11MB, indexed card metadata snapshot)
- `models/siglip2-base.onnx` (~356MB, vision encoder)

**Images bucket** (public via CloudFront `/scans/*`):
- `scans/{card_id}.jpg` (rectified user photos, ~50KB each)
- Image URLs contain UUIDs, making them effectively unguessable

scan-card downloads all three files from the data bucket on cold start.

### API Gateway

| Property | Value |
|----------|-------|
| Type | REST API (required for streaming SSR) |
| Endpoint | Regional |
| Transfer mode | STREAM |
| Integration | Lambda proxy (Nitro handler) |
| Auth | None at Gateway level (no auth) |

REST API (not HTTP API v2) required for response streaming support per the
TanStack Start deployment reference.

### Cost Estimate (Low Traffic)

| Component | Usage | Monthly Cost |
|-----------|-------|-------------|
| Nitro Lambda | 10,000 page loads, 512MB | ~$0.05 |
| Backend Lambdas (CRUD) | 5,000 calls, 128MB | ~$0.02 |
| scan-card Lambda | 1,000 scans, 2GB | ~$0.35 |
| REST API Gateway | 15,000 requests | ~$0.05 |
| DynamoDB | 1,000 cards, on-demand | ~$0 (free tier) |
| S3 storage | 1,000 photos + static assets | ~$0.05 |
| CloudFront | Static + API routing | ~$0 (free tier) |
| Route 53 | 1 hosted zone | $0.50 |
| **Total** | | **~$1.02/mo** |

At 10,000 scans/month: ~$5/month. Each operation scales independently.

### Cold Start Mitigation

| Strategy | Nitro Lambda | scan-card Lambda | CRUD Lambdas | Cost |
|----------|-------------|------------------|-------------|------|
| No mitigation | ~1-2s | ~3-5s (S3 download ~700MB) | ~5-10ms | $0 |
| CloudWatch ping every 5 min | Rare cold starts | Rare cold starts | N/A (already fast) | ~$0.10/mo |
| Provisioned Concurrency | None | None | N/A | ~$7/mo |

No warm-keeping: low-traffic personal app, cold starts are acceptable.
Rust CRUD Lambdas have near-zero cold starts (~5-10ms). scan-card cold start
(~1-3s for S3 index download) is infrequent and tolerable for a scan operation.

### Model Export

**YOLO11n-OBB** (browser): Export trained YOLO model to ONNX format (~6MB).
Served via CloudFront, cached by service worker. Same model as Phase 5.

**SigLIP2 Base p16-384** (scan-card Lambda): Already validated in Phase 4
Experiment 6. Export vision encoder + L2 normalize wrapper to ONNX with
dynamic batch axis. ~356MB FP32. INT8 quantization not recommended (Phase 4
Experiment 6). Stored in S3 data bucket.

**FAISS Index**: Pre-built with SigLIP2 Base p16-384 embeddings: 331 MB,
107,782 vectors, 768-dim. Stored in S3 data bucket. Updated by cron-update
Lambda every 2 weeks.

### Index Update Strategy

A `cron-update` Lambda runs every 2 weeks via EventBridge schedule:
1. Download latest Scryfall bulk data
2. Rebuild cards.parquet (filter playable, detect placeholders)
3. Embed new cards, rebuild FAISS index
4. Write updated card_index.faiss + card_metadata.parquet to S3
5. scan-card and search-cards pick up new files on next cold start
6. No Lambda redeployment needed

Total time: ~5 min for 300 new cards on GPU, ~20 min on CPU.

## Experiments

Planned:
- Rust ONNX Runtime inference accuracy validation (match Python output)
- Graviton ARM64 inference latency benchmarks
- Cold start measurements across all Lambda types

## Validation

Validation framework for deployment:

| Test | Method | Target |
|------|--------|--------|
| Accuracy | Rust ONNX output vs Python ONNX output (cosine similarity) | >0.999 |
| scan-card latency | CloudWatch duration metric (warm) | <2s per card |
| CRUD latency | CloudWatch duration metric (warm) | <100ms |
| Cold start | CloudWatch init duration | Nitro <2s, scan-card <5s (S3 download), CRUD <15ms |
| Page load | Lighthouse mobile score | >90 |
| Concurrent scan | 9 parallel scan-card invocations (binder page) | All return <3s |
| End-to-end | Browser scan -> library update -> grid refresh | <5s total |

## Checklist

### Frontend (Cloud Mode)
- [ ] Switch Nitro preset from `node-server` (Phase 5) to `aws-lambda`
- [ ] Configure Nitro aws-lambda preset with streaming
- [ ] Replace `fetch('http://127.0.0.1:8080/...')` calls in server functions with AWS SDK Lambda invoke
- [ ] Build and deploy Nitro bundle to Lambda

### Model Exports
- [x] Export SigLIP2 Base p16-384 to ONNX -- validated in Phase 4
- [x] Build FAISS-CPU index with SigLIP2 embeddings -- done (107,782 vectors)
- [ ] Upload siglip2-base.onnx to S3 data bucket
- [ ] Upload card_index.faiss + card_metadata.parquet to S3 data bucket

### Infrastructure (CDK)
- [x] DynamoDB table (card_id PK, GSI on scryfall_id) -- DataStack
- [x] S3 data bucket (private, Lambda-only: FAISS index, card_metadata.parquet, ONNX model) -- DataStack
- [x] S3 assets bucket (private, served via CloudFront `/assets/*`) -- CdnStack
- [x] S3 images bucket (private, served via CloudFront `/scans/*`) -- CdnStack
- [x] CloudFront distribution (S3 assets + S3 images origins, OAC) -- CdnStack
- [ ] CloudFront `/*` fallback origin pointing at REST API Gateway
- [ ] S3 lifecycle rules for scan photo retention
- [ ] REST API Gateway with streaming (for Nitro SSR)
- [ ] Nitro Lambda (Node.js 22, ARM64, 512MB)
- [x] 7 Backend Lambdas (all provided.al2023, ARM64, scan-card with 1GB ephemeral storage) -- ApiStack
- [x] 1 Cron Lambda (Rust, EventBridge schedule every 2 weeks) -- ApiStack
- [x] IAM roles (each backend Lambda -> DynamoDB/S3 as needed) -- ApiStack
- [ ] Route 53 + ACM certificate (us-east-1)
- [~] CloudWatch EventBridge ping -- not needed (low-traffic app, cold starts acceptable)

### Backend Lambdas (Rust)
- [x] Define `CardStore`, `ImageStore`, `IndexStore` traits in shared crate (`store/traits.rs`)
- [x] Implement `DynamoCardStore` (`store/aws.rs`)
- [x] Implement `S3ImageStore` + `S3IndexStore` (`store/aws.rs`)
- [x] Extract handler logic into shared functions (take `impl CardStore + ImageStore`) (`handlers/*.rs`)
- [x] Refactor Lambda crates to use traits + shared handlers (all 8 Lambdas)
- [x] Set up Rust workspace with cargo-lambda and shared types crate
- [x] All 8 Lambda handlers implemented (Phase 5 proved the shared handler logic works)
- [ ] Deploy all 8 Lambdas via `cargo-lambda deploy` + CDK integration
- [ ] End-to-end cloud test: deployed stack handles real scans

### Validation
- [ ] Cold start: measure Nitro + scan-card + CRUD Lambdas separately
- [ ] Warm latency: page load, scan round-trip, library CRUD
- [ ] End-to-end: scan from browser -> Nitro -> scan-card -> DynamoDB -> library view
- [ ] Multi-card: binder page (9 concurrent scan-card Lambda invocations)

## Reference Implementation

Architecture pattern based on: https://johanneskonings.dev/blog/2025-11-30-tanstack-start-aws-serverless/
- TanStack Start + Nitro + aws-lambda preset
- REST API Gateway with streaming
- CloudFront + S3 for static assets
- CDK infrastructure

## Conclusion

Phase 6 takes the Phase 5 deliverable (fully working local Docker container)
and lifts it to AWS serverless infrastructure. The shared Rust handler code
is unchanged between the two phases -- storage implementations differ only
in which trait implementations are wired in at binary compile time.

The frontend changes are minimal: swap the Nitro preset and repoint server
functions from localhost HTTP to Lambda IAM invoke. No React component,
route, or TanStack Query change is needed.

**CDK stacks** (synthesize successfully):
- **DataStack**: DynamoDB table + GSI, S3 data bucket (private)
- **CdnStack**: S3 assets bucket, S3 images bucket, CloudFront distribution with OAC
- **ApiStack**: 7 backend Lambdas + 1 cron Lambda + EventBridge + IAM

**Remaining work**: Nitro Lambda + aws-lambda preset, REST API Gateway,
CloudFront `/*` -> API Gateway origin, Route 53 + ACM, S3 lifecycle rules,
Lambda deployment via CDK + cargo-lambda integration, end-to-end validation.
