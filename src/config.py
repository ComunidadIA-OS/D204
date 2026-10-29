"""Loads config.yaml. If a section, a required key or the API key is missing, it raises a clear error.

The `paths` are returned as absolute paths from the repository root, so the scripts work from any
folder. The API key is never in config.yaml: it is read from the environment variable named in
llm.api_key_env.
"""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

# required keys per section; the others (llm.ca_cert, llm.timeout_seconds...) have a default value
REQUIRED = {
    # max_tokens is required: without a limit, a reasoning model may never finish
    "llm": ["base_url", "model", "api_key_env", "temperature", "max_tokens"],
    "scrape": ["start_url", "delay_seconds", "user_agent"],
    "sample": ["first_year", "last_year", "per_year", "buffer", "seed"],
    "analysis": ["model", "run", "rounds"],
    "paths": [
        "index", "year_bounds", "sample", "corpus", "articles", "raw", "fetch_errors",
        "mentions", "cache", "annotations", "metrics", "figures",
    ],
}


class ConfigError(Exception):
    pass


def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise ConfigError(f"No existe el fichero de configuración: {path}")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ConfigError(f"{path} está vacío o no es un mapa YAML")

    for section, keys in REQUIRED.items():
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"Falta la sección '{section}' en {path}")
        for key in keys:
            if config[section].get(key) is None:
                raise ConfigError(f"Falta la clave '{section}.{key}' en {path}")

    config["paths"] = {name: ROOT / p for name, p in config["paths"].items()}
    return config


def get_api_key(config: dict) -> str:
    env_var = config["llm"]["api_key_env"]
    key = os.environ.get(env_var)
    if not key:
        raise ConfigError(f"La variable de entorno {env_var} no está definida (API key del LLM)")
    return key
