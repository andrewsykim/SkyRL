import argparse
import ray
import uvicorn
from skyrl.tinker.api import app
from skyrl.tinker.config import EngineConfig, add_model
from skyrl.utils.log import get_uvicorn_log_config, logger

@ray.remote
class TinkerApiActor:
    """Ray actor that wraps the SkyRL Tinker API FastAPI application."""

    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config

    def run(self, host: str, port: int):
        """Starts the Tinker API server using uvicorn."""
        logger.info(f"Starting Tinker API server as Ray actor on {host}:{port}")
        # Store config in app.state so lifespan can access it
        app.state.engine_config = self.engine_config

        # Start uvicorn server. This blocks until the server is shut down.
        uvicorn.run(app, host=host, port=port, log_config=get_uvicorn_log_config())


def main():
    """Main entry point for running SkyRL Tinker API on Ray."""
    parser = argparse.ArgumentParser(description="SkyRL Ray Tinker API server")
    add_model(parser, EngineConfig)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--ray-address", type=str, default=None, help="Ray address to connect to")
    args = parser.parse_args()

    ray.init(address=args.ray_address, ignore_reinit_error=True)

    # Create EngineConfig from parsed arguments
    engine_config = EngineConfig.model_validate({k: v for k, v in vars(args).items() if k in EngineConfig.model_fields})

    # Configure actor resources if specified in EngineConfig
    actor_options = {}
    if engine_config.ray_actor_options:
        if "num_cpus" in engine_config.ray_actor_options:
            actor_options["num_cpus"] = engine_config.ray_actor_options["num_cpus"]
        if "num_gpus" in engine_config.ray_actor_options:
            actor_options["num_gpus"] = engine_config.ray_actor_options["num_gpus"]
        if "resources" in engine_config.ray_actor_options:
            actor_options["resources"] = engine_config.ray_actor_options["resources"]

    # Start the Tinker API actor
    logger.info("Launching TinkerApiActor...")
    actor = TinkerApiActor.options(**actor_options).remote(engine_config)

    # Run the actor and wait for it to finish
    try:
        ray.get(actor.run.remote(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down Ray actor...")


if __name__ == "__main__":
    main()
