# Qdrant Vector Database Deployment Guide

This guide covers deploying Qdrant vector database for Echo Roots semantic processing.

## Quick Start

### Local Development with Docker

1. **Start Qdrant with Docker:**
```bash
# Start Qdrant container
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant:latest

# Or use docker-compose (see docker-compose.yml below)
docker-compose up -d qdrant
```

2. **Verify Qdrant is running:**
```bash
curl http://localhost:6333/health
# Should return: {"status":"ok"}
```

3. **Initialize Echo Roots with Qdrant:**
```python
from echo_roots.storage.hybrid_manager import HybridStorageManager

# Create hybrid manager with Qdrant enabled
config = {
    "qdrant": {
        "host": "localhost",
        "port": 6333,
        "enabled": True
    }
}

manager = HybridStorageManager(config)
await manager.initialize()
```

## Docker Compose Configuration

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: echo-roots-qdrant
    ports:
      - "6333:6333"  # REST API
      - "6334:6334"  # gRPC API
    volumes:
      - ./qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  echo-roots:
    build: .
    container_name: echo-roots-app
    depends_on:
      qdrant:
        condition: service_healthy
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    volumes:
      - ./config:/app/config
      - ./data:/app/data
```

## Production Deployment

### Qdrant Cloud (Recommended)

1. **Sign up for Qdrant Cloud:**
   - Visit https://cloud.qdrant.io/
   - Create cluster and get API key

2. **Configure Echo Roots:**
```yaml
# config/qdrant_config.yaml
vector_storage:
  qdrant:
    host: "your-cluster.cloud.qdrant.io"
    port: 6333
    https: true
    api_key: "${QDRANT_API_KEY}"
```

3. **Set environment variables:**
```bash
export QDRANT_API_KEY="your-api-key-here"
```

### Self-Hosted Production

#### Option 1: Docker with Persistent Storage

```bash
# Create data directory
mkdir -p /opt/qdrant/storage

# Run with production settings
docker run -d \
  --name qdrant-prod \
  --restart unless-stopped \
  -p 6333:6333 \
  -p 6334:6334 \
  -v /opt/qdrant/storage:/qdrant/storage \
  -e QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=32 \
  -e QDRANT__SERVICE__MAX_WORKERS=4 \
  -e QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=4 \
  qdrant/qdrant:latest
```

#### Option 2: Kubernetes Deployment

```yaml
# qdrant-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qdrant
spec:
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:latest
        ports:
        - containerPort: 6333
        - containerPort: 6334
        volumeMounts:
        - name: qdrant-storage
          mountPath: /qdrant/storage
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: QDRANT__SERVICE__MAX_WORKERS
          value: "4"
      volumes:
      - name: qdrant-storage
        persistentVolumeClaim:
          claimName: qdrant-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant-service
spec:
  selector:
    app: qdrant
  ports:
  - name: http
    port: 6333
    targetPort: 6333
  - name: grpc
    port: 6334
    targetPort: 6334
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: qdrant-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
```

## Configuration Options

### Collection Configuration

Echo Roots automatically creates these collections:

1. **semantic_candidates** - Main collection for semantic candidate embeddings
2. **taxonomy_concepts** - Collection for taxonomy concept embeddings

### Performance Tuning

#### Memory Optimization

For large datasets, enable quantization:

```yaml
# config/qdrant_config.yaml
vector_storage:
  qdrant:
    collections:
      semantic_candidates:
        quantization_config:
          scalar:
            type: "int8"
            quantile: 0.99
            always_ram: true
```

#### Storage Optimization

For reduced memory usage:

```yaml
vector_storage:
  qdrant:
    collections:
      semantic_candidates:
        on_disk_payload: true
        hnsw_config:
          on_disk: true
```

#### Search Performance

For faster searches:

```yaml
vector_storage:
  qdrant:
    collections:
      semantic_candidates:
        hnsw_config:
          m: 32  # Higher = better recall, more memory
          ef_construct: 400  # Higher = better index quality
```

## Monitoring and Maintenance

### Health Checks

```bash
# Basic health check
curl http://localhost:6333/health

# Detailed cluster info
curl http://localhost:6333/cluster

# Collection stats
curl http://localhost:6333/collections/semantic_candidates
```

### Backup and Recovery

```bash
# Create snapshot
curl -X POST http://localhost:6333/collections/semantic_candidates/snapshots

# List snapshots
curl http://localhost:6333/collections/semantic_candidates/snapshots

# Download snapshot
curl http://localhost:6333/collections/semantic_candidates/snapshots/snapshot_name \
  --output backup.snapshot
```

### Performance Monitoring

Monitor these metrics:
- Memory usage (`/metrics` endpoint)
- Query latency
- Index building progress
- Collection size and segment count

## Troubleshooting

### Common Issues

1. **Connection refused:**
   - Check if Qdrant is running: `docker ps`
   - Verify port mapping: `netstat -tulpn | grep 6333`

2. **Out of memory:**
   - Reduce batch sizes in config
   - Enable quantization
   - Move payloads to disk

3. **Slow searches:**
   - Increase `ef` parameter in search requests
   - Optimize HNSW parameters
   - Consider using payload indexing

4. **Collection not found:**
   - Ensure Echo Roots has created collections
   - Check collection names match configuration

### Logging

Enable debug logging:

```yaml
# config/qdrant_config.yaml
performance:
  logging:
    level: "DEBUG"
    log_queries: true
```

### Reset Collections

```python
# In Echo Roots
from echo_roots.storage.qdrant_backend import QdrantBackend

backend = QdrantBackend(config)
await backend.initialize()

# Delete and recreate collection
await backend.delete_collection("semantic_candidates")
await backend.create_collection("semantic_candidates", vector_size=384)
```

## Security

### Network Security

1. **Firewall Configuration:**
```bash
# Only allow local connections
ufw allow from 127.0.0.1 to any port 6333
ufw allow from 127.0.0.1 to any port 6334
```

2. **API Key Authentication:**
```yaml
# Qdrant config
service:
  api_key: "your-secure-api-key"
```

### Data Encryption

For sensitive data:
- Use HTTPS/TLS for connections
- Encrypt storage volumes
- Use Qdrant Cloud for managed security

## Migration

### From DuckDB-only Setup

1. Keep existing DuckDB data
2. Enable Qdrant in configuration
3. Run embedding pipeline to populate vectors
4. Test hybrid functionality
5. Monitor performance improvements

### Between Qdrant Versions

1. Create snapshots before upgrade
2. Stop Echo Roots application
3. Upgrade Qdrant
4. Restart and verify collections
5. Restore from snapshot if needed

## Integration Examples

### Python Integration

```python
from echo_roots.semantic.pipeline import SemanticPipelineFactory
from echo_roots.storage.qdrant_backend import QdrantBackend

# Initialize Qdrant backend
qdrant_config = {"host": "localhost", "port": 6333}
qdrant_backend = QdrantBackend(qdrant_config)
await qdrant_backend.initialize()

# Create semantic pipeline
pipeline = await SemanticPipelineFactory.create_default_pipeline(
    vector_repository=qdrant_backend.semantic_repository
)

# Process candidates
await pipeline.process_semantic_candidates(candidates)
```

### API Integration

```python
from echo_roots.api import create_app

# API automatically uses configured Qdrant backend
app = create_app()

# Semantic search endpoint will use Qdrant
# GET /api/v1/semantic/search?q=electronics&limit=10
```

## Support

For issues:
1. Check logs: `docker logs qdrant-container`
2. Verify configuration files
3. Test with minimal dataset
4. Consult Qdrant documentation: https://qdrant.tech/documentation/
5. Check Echo Roots integration tests
