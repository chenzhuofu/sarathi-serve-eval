#!/bin/bash
python main.py \
    --request_generator_config_type TRACE \
    --trace_request_generator_config_date 2023-09-21 \
    --trace_request_generator_config_trace_file data/emission_trace.csv \
    --trace_request_generator_config_max_tokens 4096 \
    --trace_request_generator_config_prefill_scale_factor 1 \
    --trace_request_generator_config_decode_scale_factor 1
