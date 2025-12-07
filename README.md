# LeetTriton

Build and run detached
```bash
docker build -t triton-dev .
docker run --gpus all -d --name triton \
  -v $(pwd):/workspace \
  triton-dev sleep infinity
```