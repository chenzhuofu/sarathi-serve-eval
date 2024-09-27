import logging
import os
import yaml

from sarathi.benchmark.benchmark_runner import BenchmarkRunnerLauncher
from sarathi.benchmark.config import BenchmarkConfig, TraceRequestGeneratorConfig
from sarathi.benchmark.constants import LOGGER_FORMAT, LOGGER_TIME_FORMAT
from sarathi.benchmark.utils.random import set_seeds
from sarathi.logger import init_logger

logger = init_logger(__name__)


config = BenchmarkConfig.create_from_cli_args()

# Override the request_generator_config to use TraceRequestGeneratorConfig
config.request_generator_config = TraceRequestGeneratorConfig(
    trace_file='./data/emission_trace.csv',  # Path to your trace CSV file
    date='2023-09-21',                      # Example date filter
    prefill_scale_factor=1.0,                # Adjust as needed
    decode_scale_factor=1.0,                 # Adjust as needed
    max_tokens=4096,                         # Max number of tokens
    time_scale_factor=0.3                    # Adjust the time scaling
)

# Create the output directory if it does not exist
os.makedirs(config.output_dir, exist_ok=True)

# Save the configuration to a YAML file for reference
with open(os.path.join(config.output_dir, "config.yaml"), "w") as f:
    yaml.dump(config.to_dict(), f)

logger.info(f"Starting benchmark with config: {config}")

# Set random seeds for reproducibility
set_seeds(config.seed)

# Set the logging level based on the config
log_level = getattr(logging, config.log_level.upper())
logging.basicConfig(
    format=LOGGER_FORMAT, level=log_level, datefmt=LOGGER_TIME_FORMAT
)

# Initialize and run the benchmark
runner = BenchmarkRunnerLauncher(config)
runner.run()

