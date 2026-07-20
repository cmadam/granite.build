set -e
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
export MODEL_PATH="{{ bindings.model.binding.path }}"
export MODEL_ID="$${BFCL_MODEL_ID}"
export EXP_NAME="$${EXPERIMENT}"
export OUTPUT_DIR="${OUTPUT_BASE_PATH}/$${EXPERIMENT}/$${BFCL_EVAL_NAME}"
export TEST_CATEGORIES="$${BFCL_TEST_CATEGORIES}"
export NUM_GPUS_GENERATE=$${EVAL_NUM_GPUS}
export NUM_GPUS_EVALUATE=$${EVAL_NUM_GPUS}
export GPU_MEMORY_UTILIZATION="0.5"
export VLLM_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SKIP_P2P_CHECK=1
export NCCL_IGNORE_DISABLED_P2P=1
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR"
bash /workspace/scripts/run-bfcl.sh 2>&1 | tee "${OUTPUT_DIR}/${EXP_NAME}-bfcl.log"
RESULT_FILE=$(find "$OUTPUT_DIR" -type f \( -name '*score*.json' -o -name '*summary*.json' -o -name 'BFCL_*.json' \) -printf '%T@ %p\n' | sort -nr | awk 'NR==1{print $2}')
if [ -z "$RESULT_FILE" ]; then echo "ERROR: no BFCL result json in $OUTPUT_DIR"; find "$OUTPUT_DIR" -maxdepth 2 -type f | head -20; exit 1; fi
echo "LLMB_ARTIFACT_ID:bfcl_results LLMB_ARTIFACT_PATH:${RESULT_FILE}"
