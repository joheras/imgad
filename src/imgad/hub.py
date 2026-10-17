"""Access to gated / private models on the Hugging Face Hub.

Some models (e.g. DINOv3, some SigLIP2 or medical checkpoints) are **gated**: you must accept their
conditions on the model page and authenticate with an access token. imgad looks for a token, in order:

1. an explicit ``token=`` argument (``HFFeatureExtractor(..., token=...)``, ``load(..., token=...)``,
   ``--hf-token-file`` in the CLI),
2. the ``HF_TOKEN`` environment variable,
3. the token stored by ``imgad hf-login`` / ``hf auth login`` (``~/.cache/huggingface/token``).

The token is **never** written to saved pipelines, configs, feature-cache keys, reports or logs.

    imgad hf-login                         # stores a token (input hidden)
    imgad hf-check facebook/dinov3-vitb16-pretrain-lvd1689m
    imgad hf-check --config configs/mvtec_benchmark.yaml   # every Hub model of a benchmark
"""

from __future__ import annotations

import contextlib
import os
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional, Union

TokenLike = Union[None, bool, str]
TOKEN_ENV = "HF_TOKEN"


class GatedModelError(RuntimeError):
    """A Hub model cannot be accessed (gated, private, wrong token...). The message says what to do."""


# ----------------------------------------------------------------------------- tokens
def read_token_file(path: Union[str, os.PathLike]) -> str:
    """Read a token from a file (first non-empty line). Warns if other users can read the file."""
    p = Path(path).expanduser()
    token = next((line.strip() for line in p.read_text().splitlines() if line.strip()), "")
    if not token:
        raise ValueError(f"No token found in {p}")
    try:
        if p.stat().st_mode & (stat.S_IRGRP | stat.S_IROTH):
            import warnings

            warnings.warn(f"{p} is readable by other users; consider `chmod 600 {p}`", stacklevel=2)
    except OSError:  # pragma: no cover
        pass
    return token


def token_source(token: TokenLike = None) -> str:
    """Where the token that will be used comes from (never the token itself)."""
    if token is False:
        return "none (disabled)"
    if isinstance(token, str):
        return "explicit token"
    if os.environ.get(TOKEN_ENV):
        return f"{TOKEN_ENV} environment variable"
    try:
        from huggingface_hub import get_token

        if get_token():
            return "stored login (hf auth login / imgad hf-login)"
    except Exception:  # pragma: no cover
        pass
    return "none"


def whoami(token: TokenLike = None) -> Optional[str]:
    """User name the token belongs to, or None if there is no valid token."""
    if token is False:
        return None
    try:
        from huggingface_hub import whoami as _whoami

        return _whoami(token=token if isinstance(token, str) else None).get("name")
    except Exception:
        return None


def login(token: Optional[str] = None) -> str:
    """Store a token like ``hf auth login`` (asks for it with hidden input if not given) and return the
    user name. Does not touch git credentials."""
    import getpass

    from huggingface_hub import login as _login

    if not token:
        token = getpass.getpass("Hugging Face access token (input hidden): ").strip()
    if not token:
        raise ValueError("No token given")
    user = whoami(token)
    if user is None:
        raise GatedModelError("The token is not valid (Hugging Face rejected it).")
    _login(token=token, add_to_git_credential=False)
    return user


def logout() -> None:
    from huggingface_hub import logout as _logout

    _logout()


@contextlib.contextmanager
def token_env(token: TokenLike):
    """Temporarily expose an explicit token as ``HF_TOKEN`` (for libraries such as open_clip or timm that
    download from the Hub but do not accept a token argument)."""
    if not isinstance(token, str):
        yield
        return
    old = os.environ.get(TOKEN_ENV)
    os.environ[TOKEN_ENV] = token
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(TOKEN_ENV, None)
        else:
            os.environ[TOKEN_ENV] = old


def pop_token_kwargs(kwargs: dict) -> TokenLike:
    """Remove ``token`` / ``use_auth_token`` from a kwargs dict (so they are never stored) and return it."""
    token = kwargs.pop("token", None)
    legacy = kwargs.pop("use_auth_token", None)
    return token if token is not None else legacy


# ----------------------------------------------------------------------------- access checks
@dataclass
class AccessStatus:
    repo_id: str
    ok: bool
    state: str                 # "public", "authorized", "cached", "gated", "not_found", "bad_token", "offline", "error"
    gated: Optional[str] = None     # "auto" | "manual" | None
    private: Optional[bool] = None
    cached: bool = False
    user: Optional[str] = None
    token_source: str = "none"
    detail: str = ""
    license: Optional[str] = None   # from the model card metadata, when available

    def explain(self) -> str:
        text = self._explain()
        if self.license and self.state in ("public", "authorized", "cached", "gated"):
            text += f" License: {self.license}."
        return text

    def _explain(self) -> str:
        url = f"https://huggingface.co/{self.repo_id}"
        if self.state == "public":
            return f"{self.repo_id}: public model, no token needed."
        if self.state == "authorized":
            return f"{self.repo_id}: gated/private model, access granted (user {self.user}, token from {self.token_source})."
        if self.state == "cached":
            return (f"{self.repo_id}: no access with the current token, but the model files are in the local Hugging Face "
                    f"cache, so it can be loaded. To refresh it, get access at {url}.")
        if self.state == "not_found":
            return (f"{self.repo_id}: not found. Check the name, or, if it is a private repo, authenticate with a token "
                    f"that can read it ({_how_to_auth()}).")
        if self.state == "bad_token":
            return f"{self.repo_id}: the token (from {self.token_source}) was rejected by Hugging Face. {_how_to_auth()}."
        if self.state == "gated":
            steps = [f"1. Open {url}, log in and accept the access conditions"
                     + (" (requests are reviewed by the authors, approval may take a while)." if self.gated == "manual" else ".")]
            if self.user:
                steps.append(f"2. You are authenticated as '{self.user}' (token from {self.token_source}); once access is "
                             "granted, run again. If you already accepted, make sure this token belongs to that account "
                             "and, for fine-grained tokens, that it can read gated repos.")
            elif self.token_source.startswith("none"):
                steps.append(f"2. Authenticate: {_how_to_auth()}.")
            else:
                steps.append(f"2. A token is set ({self.token_source}) but Hugging Face rejected it: create a new read "
                             "token at https://huggingface.co/settings/tokens and use it (`imgad hf-login`).")
            return f"{self.repo_id} is a gated model and you do not have access yet:\n  " + "\n  ".join(steps)
        if self.state == "offline":
            return (f"{self.repo_id}: offline mode (HF_HUB_OFFLINE=1) and the model is not in the local cache; "
                    "it cannot be loaded.")
        return f"{self.repo_id}: could not check access ({self.detail})."


def _how_to_auth() -> str:
    return (f"run `imgad hf-login` (or `hf auth login`), or set the {TOKEN_ENV} environment variable, or pass "
            "--hf-token-file / token=... (create a read token at https://huggingface.co/settings/tokens)")


def is_hub_model(name: object) -> bool:
    """``org/model`` ids (also ``hf-hub:org/model`` for open_clip); not local paths."""
    if not isinstance(name, str):
        return False
    name = name[len("hf-hub:"):] if name.startswith("hf-hub:") else name
    return name.count("/") == 1 and not name.startswith(("/", ".", "~")) and not Path(name).exists()


def _repo(name: str) -> str:
    return name[len("hf-hub:"):] if name.startswith("hf-hub:") else name


def is_cached(repo_id: str) -> bool:
    """True if the local HF cache has weights for the repo (``*.safetensors`` / ``*.bin``)."""
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id == repo_id and repo.repo_type == "model":
                return any(f.file_name.endswith((".safetensors", ".bin")) for rev in repo.revisions for f in rev.files)
    except Exception:
        pass
    return False


def _license(info) -> Optional[str]:
    cd = getattr(info, "card_data", None)
    lic = None
    if cd is not None:
        lic = cd.get("license") if hasattr(cd, "get") else getattr(cd, "license", None)
        if lic == "other":
            lic = (cd.get("license_name") if hasattr(cd, "get") else None) or "other (see model card)"
    if lic is None:
        lic = next((t.split(":", 1)[1] for t in (getattr(info, "tags", None) or []) if t.startswith("license:")), None)
    return lic


def check_access(repo_id: str, token: TokenLike = None) -> AccessStatus:
    """Can this model be downloaded/loaded with the current credentials (and under which license, when the
    model card declares it)? Never raises."""
    return _check_access(repo_id, token)


def _check_access(repo_id: str, token: TokenLike = None) -> AccessStatus:
    repo_id = _repo(repo_id)
    src = token_source(token)
    cached = is_cached(repo_id)
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return AccessStatus(repo_id, cached, "cached" if cached else "offline", cached=cached, token_source=src)
    try:
        from huggingface_hub import HfApi, auth_check
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError as e:  # pragma: no cover
        return AccessStatus(repo_id, cached, "error", cached=cached, token_source=src, detail=str(e))
    tok = token if token is not None else None
    gated = private = lic = None
    try:
        info = HfApi().model_info(repo_id, token=tok, timeout=15)
        gated = info.gated or None
        private = info.private
        lic = _license(info)
    except GatedRepoError:
        gated = "unknown"
    except RepositoryNotFoundError:
        pass
    except Exception as e:
        if _status(e) == 401:
            return AccessStatus(repo_id, cached, "cached" if cached else "bad_token", cached=cached, token_source=src)
        return AccessStatus(repo_id, cached, "cached" if cached else "error", cached=cached, token_source=src, detail=_short(e))
    try:
        auth_check(repo_id, token=tok)
    except GatedRepoError:
        user = whoami(token)
        state = "cached" if cached else "gated"
        return replace(AccessStatus(repo_id, cached, state, gated if gated != "unknown" else "manual", private, cached, user, src), license=lic)
    except RepositoryNotFoundError:
        return replace(AccessStatus(repo_id, cached, "cached" if cached else "not_found", gated, private, cached, whoami(token), src), license=lic)
    except Exception as e:
        if _status(e) == 401:
            return replace(AccessStatus(repo_id, cached, "cached" if cached else "bad_token", gated, private, cached, None, src), license=lic)
        return replace(AccessStatus(repo_id, cached, "cached" if cached else "error", gated, private, cached, None, src, _short(e)), license=lic)
    if gated or private:
        return replace(AccessStatus(repo_id, True, "authorized", gated, private, cached, whoami(token), src), license=lic)
    return replace(AccessStatus(repo_id, True, "public", None, private, cached, None, src), license=lic)


def _status(exc: BaseException) -> Optional[int]:
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None)


def _short(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"


def hub_error(exc: BaseException) -> Optional[str]:
    """If ``exc`` (or its causes) is an authentication/gating error from the Hub, return its kind."""
    try:
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:  # pragma: no cover
        return None
    seen, c = set(), exc
    while c is not None and id(c) not in seen:
        seen.add(id(c))
        if isinstance(c, GatedRepoError):
            return "gated"
        if isinstance(c, RepositoryNotFoundError):
            return "not_found"
        if _status(c) == 401:
            return "unauthorized"
        c = c.__cause__ or c.__context__
    msg = str(exc).lower()
    if "gated repo" in msg:
        return "gated"
    return None


def raise_if_hub_error(exc: BaseException, repo_id: str, token: TokenLike = None) -> None:
    """Re-raise a loading error as :class:`GatedModelError` with instructions when it is an access problem."""
    if hub_error(exc) is not None:
        raise GatedModelError(check_access(repo_id, token).explain()) from exc


def check_models(names: Iterable[str], token: TokenLike = None) -> List[AccessStatus]:
    """Access status of every Hub model in ``names`` (non-Hub names are skipped)."""
    seen, out = set(), []
    for n in names:
        if is_hub_model(n) and _repo(n) not in seen:
            seen.add(_repo(n))
            out.append(check_access(n, token))
    return out
