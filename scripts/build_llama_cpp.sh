#!/usr/bin/env bash
# llama.cpp 빌드 (코랩용). 사용법: bash scripts/build_llama_cpp.sh /content/llama.cpp [캐시폴더]
# 캐시폴더(예: 구글 드라이브)에 빌드 결과가 있으면 다시 빌드하지 않고 복사만 한다.
set -euo pipefail

DIR=${1:-/content/llama.cpp}
CACHE=${2:-}
COMMIT=493253fe5a5af7e800b3c92a2452b5684a99e961   # 2026-10-21 확인한 버전 (gpt-oss 변환/MXFP4 지원)

# LLAMA_BACKEND=cpu 로 주면 GPU가 있어도 CPU 빌드 (변환/양자화만 할 때 빠름)
BACKEND=${LLAMA_BACKEND:-}
if [ -z "$BACKEND" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then BACKEND=cuda; else BACKEND=cpu; fi
fi
TAG="$BACKEND-${COMMIT:0:7}"

if [ ! -d "$DIR/.git" ]; then
  git clone -q https://github.com/ggml-org/llama.cpp "$DIR"
fi
git -C "$DIR" fetch -q --depth 1 origin "$COMMIT" 2>/dev/null || true
git -C "$DIR" checkout -q "$COMMIT"

# 변환 스크립트용 파이썬 패키지 (torch는 코랩에 있는 것을 그대로 씀)
pip install -q -e "$DIR/gguf-py" sentencepiece protobuf

if [ -n "$CACHE" ] && [ -f "$CACHE/llama-bin-$TAG.tar" ]; then
  mkdir -p "$DIR/build/bin" && tar -xf "$CACHE/llama-bin-$TAG.tar" -C "$DIR/build/bin"
  echo "캐시에서 복사함: $CACHE/llama-bin-$TAG.tar"
  exit 0
fi

FLAGS="-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF"
# T4(75), A100(80), L4(86/89), H100(90) 만 빌드해서 컴파일 시간 단축
[ "$BACKEND" = cuda ] && FLAGS="$FLAGS -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75;80;89;90"
cmake -S "$DIR" -B "$DIR/build" $FLAGS -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$DIR/build" -j"$(nproc)" --target llama-server llama-bench llama-perplexity llama-quantize

if [ -n "$CACHE" ]; then
  mkdir -p "$CACHE"
  tar -cf "$CACHE/llama-bin-$TAG.tar" -C "$DIR/build/bin" .
fi
echo "완료: $DIR/build/bin ($BACKEND)"
