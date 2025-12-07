# LeetTriton

Build and run detached based on NVIDIA pytorch
```bash
docker build -t triton-dev .
docker run --gpus all -d --name triton \
  -v $(pwd):/workspace \
  triton-dev sleep infinity
```

Lean version on pure CUDA image with torch and triton server
```bash
docker build -f Triton.dockerfile -t triton-lean .
docker run --gpus all -d --name triton-lean \
  -v $(pwd):/workspace \
  triton-lean sleep infinity
```