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

"""Main entry point for the Ollama-compatible LLM inference server."""
import logging
import sys
import os
from typing import Optional

from flask import Flask

from .config import parse_arguments
from .utils.logging import setup_logging
from .models.manager import ModelManager
from .core.request_tracker import RequestTracker
from .core.executor import LLAMAExecutor, find_llama_executable
from .api.routes import create_routes

logger = logging.getLogger(__name__)


def create_app(config) -> tuple[Flask, ModelManager, RequestTracker, Optional[LLAMAExecutor]]:
    """Create and configure the Flask application."""
    
    # Setup logging
    setup_logging(config.log_dir, config.debug)
    
    # Initialize Flask app with template folder
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    app = Flask(__name__, template_folder=template_dir)
    app.config['DEBUG'] = config.debug
    
    # Initialize core components
    logger.info("Initializing core components...")
    
    # Model Manager
    model_manager = ModelManager(config.models_base_dir)
    logger.info(f"Model manager initialized with base dir: {config.models_base_dir}")
    
    # Request Tracker
    request_tracker = RequestTracker(model_manager)
    logger.info("Request tracker initialized")
    
    # LLAMA Executor
    llama_executor = None
    try:
        llama_cli_path = find_llama_executable(config.llama_cpp_dir)
        llama_executor = LLAMAExecutor(llama_cli_path, model_manager, request_tracker)
        logger.info(f"LLAMA executor initialized with CLI: {llama_cli_path}")
    except Exception as e:
        logger.critical(f"Failed to initialize LLAMA executor: {e}", exc_info=True)
        logger.warning("Server will start but model inference will not be available")
    
    # Register routes
    api_blueprint = create_routes(model_manager, request_tracker, llama_executor, config)
    app.register_blueprint(api_blueprint)
    logger.info("API routes registered")
    
    # Test model discovery
    try:
        model_mapping = model_manager.build_model_mapping()
        logger.info(f"Discovered {len(model_mapping)} models")
        if model_mapping:
            sample_models = list(model_mapping.keys())[:3]
            logger.info(f"Sample models: {sample_models}")
        else:
            logger.warning("No models found! Check your model directory configuration.")
    except Exception as e:
        logger.error(f"Error during model discovery: {e}")
    
    return app, model_manager, request_tracker, llama_executor


def main():
    """Main entry point."""
    try:
        # Parse configuration
        config = parse_arguments()
        logger.info(f"Starting LLM Inference Service on {config.host}:{config.port}")
        logger.info(f"Configuration: models={config.model_dir}, llama.cpp={config.llama_cpp_dir}")
        
        # Create application
        app, model_manager, request_tracker, llama_executor = create_app(config)
        
        # Start the server
        logger.info("=" * 60)
        logger.info("LLM Inference Service Starting")
        logger.info("=" * 60)
        logger.info(f"🚀 Server starting on http://{config.host}:{config.port}")
        logger.info(f"📊 Dashboard available at http://{config.host}:{config.port}/dashboard")
        logger.info(f"❤️  Health check at http://{config.host}:{config.port}/health")
        logger.info("=" * 60)
        
        # Run the Flask development server
        app.run(
            host=config.host,
            port=config.port,
            debug=config.debug,
            threaded=True,
            use_reloader=False  # Disable reloader to prevent issues with threading
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.critical(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()