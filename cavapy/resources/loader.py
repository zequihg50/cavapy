import json
from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=None)
def load_json(name: str):
    """Load and cache a JSON resource from this package."""
    resource = files("cavapy.resources").joinpath(name)

    with resource.open("r", encoding="utf-8") as file:
        return json.load(file)
