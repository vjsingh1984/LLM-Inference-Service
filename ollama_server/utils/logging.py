# Copyright 2025 LLM Inference Service Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logging configuration for the application."""
import logging
import sys
from pathlib import Path


def setup_logging(log_dir: Path, debug: bool = False) -> None:
    """Setup application logging configuration.
    
    Args:
        log_dir: Directory to store log files
        debug: Enable debug level logging
    """
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'server.log'),
            logging.StreamHandler(sys.stdout)  # Ensure logs go to stdout for container environments
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Level: {logging.getLevelName(level)}, Log dir: {log_dir}")