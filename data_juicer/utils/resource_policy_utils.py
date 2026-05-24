import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urljoin

from loguru import logger

from .cache_utils import (
    DATA_JUICER_ASSETS_CACHE,
    DATA_JUICER_EXTERNAL_MODELS_HOME,
    DATA_JUICER_MODELS_CACHE,
)

_FALSE_VALUES = {"0", "false", "no", "off"}
_TRUE_VALUES = {"1", "true", "yes", "on"}

DEFAULT_RESOURCE_BASE_URL = "https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer"
DEFAULT_MODEL_BASE_URL = f"{DEFAULT_RESOURCE_BASE_URL}/models"
DEFAULT_ASSET_BASE_URL = DEFAULT_RESOURCE_BASE_URL
DEFAULT_ASSET_URLS = {
    "flagged_words": f"{DEFAULT_ASSET_BASE_URL}/flagged_words.json",
    "stopwords": f"{DEFAULT_ASSET_BASE_URL}/stopwords.json",
}

RESOURCE_KIND_LOCAL_PATH = "local_path"
RESOURCE_KIND_REMOTE_URL = "remote_url"

RESOURCE_ORIGIN_EXPLICIT_PATH = "explicit_path"
RESOURCE_ORIGIN_CACHE = "cache"
RESOURCE_ORIGIN_EXTERNAL_ROOT = "external_root"
RESOURCE_ORIGIN_LOCAL_CACHE_ROOT = "local_cache_root"
RESOURCE_ORIGIN_MIRROR = "mirror"
RESOURCE_ORIGIN_DEFAULT_PUBLIC = "default_public"


class ResourcePolicyError(ValueError):
    """Raised when a resource policy environment variable is invalid."""


class ResourceResolutionError(RuntimeError):
    """Raised when resource resolution fails under the current policy."""


@dataclass(frozen=True)
class ResourceLocation:
    kind: str
    uri: str
    origin: str

    @property
    def is_local(self) -> bool:
        return self.kind == RESOURCE_KIND_LOCAL_PATH

    @property
    def is_remote(self) -> bool:
        return self.kind == RESOURCE_KIND_REMOTE_URL

    @property
    def is_mirror(self) -> bool:
        return self.origin == RESOURCE_ORIGIN_MIRROR


def _get_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _parse_bool_env(name: str, default: Optional[bool] = None) -> Optional[bool]:
    raw = _get_env(name)
    if raw is None:
        return default

    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ResourcePolicyError(f"Invalid boolean value for {name}: {raw}")


def _parse_list_env(name: str) -> List[str]:
    raw = _get_env(name)
    if raw is None:
        return []
    return [item.strip() for item in raw.split(os.pathsep) if item.strip()]


def join_resource_url(base_url: str, *parts: str) -> str:
    url = base_url.rstrip("/") + "/"
    for part in parts:
        url = urljoin(url, str(part).lstrip("/"))
    return url


def get_resource_policy() -> Dict[str, object]:
    offline_mode = _parse_bool_env("DJ_RESOURCE_OFFLINE_MODE", default=False)
    allow_public_fallback = _parse_bool_env("DJ_RESOURCE_ALLOW_PUBLIC_FALLBACK", default=True)
    hf_local_files_only = _parse_bool_env("DJ_HF_LOCAL_FILES_ONLY", default=None)
    nltk_allow_download = _parse_bool_env("DJ_NLTK_ALLOW_DOWNLOAD", default=True)
    package_auto_install = _parse_bool_env("DJ_PACKAGE_AUTO_INSTALL", default=True)
    resource_base_url = _get_env("DJ_RESOURCE_BASE_URL")
    model_base_url = _get_env("DJ_MODEL_BASE_URL")
    asset_base_url = _get_env("DJ_ASSET_BASE_URL")

    if resource_base_url:
        normalized_resource_base_url = resource_base_url.rstrip("/")
        model_base_url = model_base_url or f"{normalized_resource_base_url}/models"
        asset_base_url = asset_base_url or normalized_resource_base_url

    if offline_mode:
        allow_public_fallback = False
        package_auto_install = False

    return {
        "offline_mode": offline_mode,
        "allow_public_fallback": allow_public_fallback,
        "local_cache_roots": _parse_list_env("DJ_RESOURCE_LOCAL_CACHE_ROOTS"),
        "resource_base_url": resource_base_url,
        "model_base_url": model_base_url,
        "asset_base_url": asset_base_url,
        "hf_endpoint": _get_env("DJ_HF_ENDPOINT"),
        "hf_home": _get_env("DJ_HF_HOME"),
        "hf_local_files_only": hf_local_files_only,
        "nltk_data_dir": _get_env("DJ_NLTK_DATA_DIR"),
        "nltk_allow_download": nltk_allow_download,
        "package_auto_install": package_auto_install,
    }


def should_allow_public_fallback(policy: Optional[Dict[str, object]] = None) -> bool:
    policy = policy or get_resource_policy()
    return bool(policy["allow_public_fallback"]) and not bool(policy["offline_mode"])


def should_auto_install_package(policy: Optional[Dict[str, object]] = None) -> bool:
    policy = policy or get_resource_policy()
    return bool(policy["package_auto_install"]) and not bool(policy["offline_mode"])


def is_nltk_download_allowed(policy: Optional[Dict[str, object]] = None) -> bool:
    policy = policy or get_resource_policy()
    return bool(policy["nltk_allow_download"]) and not bool(policy["offline_mode"])


def _find_local_model_path(model_name: str, policy: Dict[str, object]) -> Optional[ResourceLocation]:
    if os.path.exists(model_name):
        return ResourceLocation(RESOURCE_KIND_LOCAL_PATH, model_name, RESOURCE_ORIGIN_EXPLICIT_PATH)

    cache_path = os.path.join(DATA_JUICER_MODELS_CACHE, model_name)
    if os.path.exists(cache_path):
        return ResourceLocation(RESOURCE_KIND_LOCAL_PATH, cache_path, RESOURCE_ORIGIN_CACHE)

    if DATA_JUICER_EXTERNAL_MODELS_HOME:
        for path in DATA_JUICER_EXTERNAL_MODELS_HOME.split(os.pathsep):
            clean_path = path.strip()
            if not clean_path:
                continue
            external_path = os.path.join(clean_path, model_name)
            if os.path.exists(external_path):
                return ResourceLocation(RESOURCE_KIND_LOCAL_PATH, external_path, RESOURCE_ORIGIN_EXTERNAL_ROOT)

    for root in policy["local_cache_roots"]:
        candidate = os.path.join(root, model_name)
        if os.path.exists(candidate):
            return ResourceLocation(RESOURCE_KIND_LOCAL_PATH, candidate, RESOURCE_ORIGIN_LOCAL_CACHE_ROOT)

    return None


def resolve_model_source(model_name: str, force: bool = False) -> ResourceLocation:
    policy = get_resource_policy()
    attempted_sources = []

    if not force:
        local_result = _find_local_model_path(model_name, policy)
        if local_result:
            logger.info(f"Resolved model [{model_name}] from {local_result.origin}: {local_result.uri}")
            return local_result
        attempted_sources.extend(
            [
                RESOURCE_ORIGIN_EXPLICIT_PATH,
                RESOURCE_ORIGIN_CACHE,
                RESOURCE_ORIGIN_EXTERNAL_ROOT,
                RESOURCE_ORIGIN_LOCAL_CACHE_ROOT,
            ]
        )

    if policy["model_base_url"]:
        mirror_url = join_resource_url(str(policy["model_base_url"]), model_name)
        logger.info(f"Resolved model [{model_name}] to mirror URL: {mirror_url}")
        return ResourceLocation(RESOURCE_KIND_REMOTE_URL, mirror_url, RESOURCE_ORIGIN_MIRROR)
    attempted_sources.append(RESOURCE_ORIGIN_MIRROR)

    if should_allow_public_fallback(policy):
        default_url = join_resource_url(DEFAULT_MODEL_BASE_URL, model_name)
        logger.info(f"Resolved model [{model_name}] to default public source: {default_url}")
        return ResourceLocation(RESOURCE_KIND_REMOTE_URL, default_url, RESOURCE_ORIGIN_DEFAULT_PUBLIC)

    raise ResourceResolutionError(
        f"Cannot resolve model [{model_name}] under current policy. attempted_sources={attempted_sources}, "
        f"offline_mode={policy['offline_mode']}, allow_public_fallback={policy['allow_public_fallback']}"
    )


def resolve_asset_source(asset_type: str) -> ResourceLocation:
    policy = get_resource_policy()
    attempted_sources = []

    cache_path = os.path.join(DATA_JUICER_ASSETS_CACHE, f"{asset_type}.json")
    if os.path.exists(cache_path):
        logger.info(f"Resolved asset [{asset_type}] from cache: {cache_path}")
        return ResourceLocation(RESOURCE_KIND_LOCAL_PATH, cache_path, RESOURCE_ORIGIN_CACHE)
    attempted_sources.append(RESOURCE_ORIGIN_CACHE)

    for root in policy["local_cache_roots"]:
        candidate = os.path.join(root, f"{asset_type}.json")
        if os.path.exists(candidate):
            logger.info(f"Resolved asset [{asset_type}] from local cache root: {candidate}")
            return ResourceLocation(RESOURCE_KIND_LOCAL_PATH, candidate, RESOURCE_ORIGIN_LOCAL_CACHE_ROOT)
    attempted_sources.append(RESOURCE_ORIGIN_LOCAL_CACHE_ROOT)

    if policy["asset_base_url"]:
        mirror_url = join_resource_url(str(policy["asset_base_url"]), f"{asset_type}.json")
        logger.info(f"Resolved asset [{asset_type}] to mirror URL: {mirror_url}")
        return ResourceLocation(RESOURCE_KIND_REMOTE_URL, mirror_url, RESOURCE_ORIGIN_MIRROR)
    attempted_sources.append(RESOURCE_ORIGIN_MIRROR)

    if should_allow_public_fallback(policy):
        if asset_type not in DEFAULT_ASSET_URLS:
            raise ResourceResolutionError(
                f"Cannot resolve asset [{asset_type}] because it is not in default asset URLs."
            )
        logger.info(f"Resolved asset [{asset_type}] to default public source")
        return ResourceLocation(
            RESOURCE_KIND_REMOTE_URL, DEFAULT_ASSET_URLS[asset_type], RESOURCE_ORIGIN_DEFAULT_PUBLIC
        )

    raise ResourceResolutionError(
        f"Cannot resolve asset [{asset_type}] under current policy. attempted_sources={attempted_sources}, "
        f"offline_mode={policy['offline_mode']}, allow_public_fallback={policy['allow_public_fallback']}"
    )


def configure_hf_env(policy: Optional[Dict[str, object]] = None) -> None:
    policy = policy or get_resource_policy()

    if policy["hf_endpoint"]:
        os.environ["HF_ENDPOINT"] = str(policy["hf_endpoint"])
    if policy["hf_home"]:
        os.environ["HF_HOME"] = str(policy["hf_home"])

    hf_local_files_only = policy["hf_local_files_only"]
    if hf_local_files_only is True or policy["offline_mode"]:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def configure_nltk_env(policy: Optional[Dict[str, object]] = None) -> None:
    policy = policy or get_resource_policy()
    nltk_data_dir = policy["nltk_data_dir"]
    if not nltk_data_dir:
        return

    import nltk

    nltk_data_dir = str(nltk_data_dir)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)
