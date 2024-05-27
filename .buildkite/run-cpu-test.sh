# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Try building the docker image
docker build -t cpu-test -f Dockerfile.cpu .

# Setup cleanup
remove_docker_container() { docker rm -f cpu-test || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image
docker run -itd -v ~/.cache/huggingface:/root/.cache/huggingface --network host --env VLLM_CPU_KVCACHE_SPACE=4 --name cpu-test cpu-test

# offline inference
docker exec cpu-test bash -c "python3 vllm/examples/offline_inference.py"

# async engine test, not passing due to distributed inference support missing
#docker exec cpu-test bash -c "cd tests; pytest -v -s async_engine"

# Run basic model test
docker exec cpu-test bash -c "cd vllm/tests;
  pip install pytest Pillow
  bash ../.buildkite/download-images.sh
  pytest -v -s models --ignore=models/test_llava.py --ignore=models/test_mistral.py \
    --ignore=models/test_big_models.py --ignore=models/test_embedding.py"

# Run big model test
#docker exec cpu-test bash -c "cd tests;
#  sed -i 's/half/float/g' models/test_big_models.py
#  pytest -v -s models/test_big_models.py"
