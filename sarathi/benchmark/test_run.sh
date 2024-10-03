#!/bin/bash
python main.py \
    --model_config_model meta-llama/Meta-Llama-3-8B-Instruct \
    --request_generator_config_type TRACE \
    --trace_request_generator_config_date 2023-09-21 \
    --trace_request_generator_config_trace_file data/emission_trace_slo.csv \
    --trace_request_generator_config_max_tokens 4096 \
    --baseline_latency_ms 100 \
    --tensor_parallel_size 1 \
    --pipeline_parallel_size 1
