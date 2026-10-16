"""Gated / private Hugging Face models. The Hub is simulated: no network or real token is needed."""

import os
import pickle
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

import imgad.hub as hub
from imgad import AnomalyDetectionPipeline, GatedModelError, HFFeatureExtractor, benchmark, data
from imgad.cli import main

SECRET = "hf_THIS_IS_A_TEST_SECRET"


def _err(cls, status=401):
    resp = httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/api/models/x"))
    return cls("error", response=resp)


@pytest.fixture
def fake_hub(monkeypatch):
    """Simulated Hub: ``repos[id] = {"gated": ..., "access": bool}``; ``users[token] = name``."""
    state = SimpleNamespace(repos={}, users={}, cached=set())
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: os.environ.get("HF_TOKEN"))

    def current(token):
        return token if isinstance(token, str) else os.environ.get("HF_TOKEN")

    def model_info(self, repo_id, token=None, timeout=None):
        if repo_id not in state.repos:
            raise _err(RepositoryNotFoundError, 404)
        r = state.repos[repo_id]
        return SimpleNamespace(gated=r.get("gated", False), private=r.get("private", False))

    def auth_check(repo_id, token=None):
        if repo_id not in state.repos:
            raise _err(RepositoryNotFoundError, 404)
        r = state.repos[repo_id]
        if r.get("gated") and not (current(token) in state.users and r.get("access")):
            raise _err(GatedRepoError)

    def whoami(token=None):
        t = current(token)
        if t not in state.users:
            raise _err(RepositoryNotFoundError, 401)
        return {"name": state.users[t]}

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", model_info)
    monkeypatch.setattr("huggingface_hub.auth_check", auth_check)
    monkeypatch.setattr("huggingface_hub.whoami", whoami)
    monkeypatch.setattr(hub, "is_cached", lambda repo_id: repo_id in state.cached)
    return state


# ----------------------------------------------------------------------------- token never stored
def test_token_is_never_stored(tmp_path):
    ext = HFFeatureExtractor("org/model", token=SECRET, hf_kwargs={"use_auth_token": "hf_OTHER", "foo": 1})
    assert ext._token == SECRET and ext.hf_kwargs == {"foo": 1}
    for text in (repr(ext), ext.identifier, str(ext.get_config())):
        assert SECRET not in text and "hf_OTHER" not in text
    assert pickle.loads(pickle.dumps(ext))._token is None

    pipe = AnomalyDetectionPipeline(ext, "knn")
    pipe.save(tmp_path / "p.joblib")
    assert SECRET.encode() not in (tmp_path / "p.joblib").read_bytes()
    loaded = AnomalyDetectionPipeline.load(tmp_path / "p.joblib", token="hf_LOAD")
    assert loaded.extractor._token == "hf_LOAD"
    assert AnomalyDetectionPipeline.load(tmp_path / "p.joblib").extractor._token is None


def test_token_env_context(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_before")
    with hub.token_env("hf_temp"):
        assert os.environ["HF_TOKEN"] == "hf_temp"
    assert os.environ["HF_TOKEN"] == "hf_before"
    with hub.token_env(None):
        assert os.environ["HF_TOKEN"] == "hf_before"


def test_read_token_file(tmp_path):
    f = tmp_path / "tok"
    f.write_text(f"\n  {SECRET}  \n")
    f.chmod(0o600)
    assert hub.read_token_file(f) == SECRET
    f.chmod(0o644)
    with pytest.warns(UserWarning, match="chmod 600"):
        hub.read_token_file(f)
    (tmp_path / "empty").write_text("\n")
    with pytest.raises(ValueError):
        hub.read_token_file(tmp_path / "empty")


# ----------------------------------------------------------------------------- access checks
def test_check_access_states(fake_hub, monkeypatch):
    fake_hub.repos = {"org/public": {}, "org/gated": {"gated": "manual", "access": True},
                      "org/noaccess": {"gated": "auto", "access": False}}
    fake_hub.users = {SECRET: "alice"}

    assert hub.check_access("org/public").state == "public"
    st = hub.check_access("org/gated")
    assert (st.ok, st.state, st.gated) == (False, "gated", "manual")
    assert "imgad hf-login" in st.explain() and "reviewed by the authors" in st.explain()
    assert hub.check_access("org/missing").state == "not_found"

    st = hub.check_access("org/gated", token=SECRET)
    assert (st.ok, st.state, st.user, st.token_source) == (True, "authorized", "alice", "explicit token")
    monkeypatch.setenv("HF_TOKEN", SECRET)
    st = hub.check_access("org/noaccess")
    assert st.state == "gated" and "authenticated as 'alice'" in st.explain()
    assert st.token_source == "HF_TOKEN environment variable"

    monkeypatch.setenv("HF_TOKEN", "hf_wrong")
    assert "rejected" in hub.check_access("org/noaccess").explain()

    fake_hub.cached.add("org/noaccess")
    st = hub.check_access("org/noaccess")
    assert st.ok and st.state == "cached"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert hub.check_access("org/noaccess").state == "cached"
    assert hub.check_access("org/gated").state == "offline"


def test_explanations_never_contain_the_token(fake_hub):
    fake_hub.repos = {"org/gated": {"gated": "auto", "access": False}}
    fake_hub.users = {SECRET: "alice"}
    assert SECRET not in hub.check_access("org/gated", token=SECRET).explain()


def test_is_hub_model(tmp_path):
    assert hub.is_hub_model("facebook/dinov2-base") and hub.is_hub_model("hf-hub:laion/CLIP-ViT-B-32")
    assert not hub.is_hub_model("ViT-B-32") and not hub.is_hub_model(str(tmp_path))


def test_loading_a_gated_model_raises_with_instructions(fake_hub, monkeypatch):
    fake_hub.repos = {"org/gated": {"gated": "manual", "access": False}}

    def from_pretrained(*a, **kw):
        raise OSError("You are trying to access a gated repo.") from _err(GatedRepoError)

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", from_pretrained)
    ext = HFFeatureExtractor("org/gated", device="cpu")
    with pytest.raises(GatedModelError, match="https://huggingface.co/org/gated"):
        ext._load()


def test_gated_but_cached_model_loads_from_cache(fake_hub, monkeypatch, caplog):
    fake_hub.repos = {"org/gated": {"gated": "manual", "access": False}}
    fake_hub.cached.add("org/gated")
    monkeypatch.setattr("imgad.features.huggingface.is_cached", lambda repo: repo in fake_hub.cached)
    calls = []

    class Fake:
        def eval(self):
            return self

        def to(self, device):
            return self

        def get_image_features(self):  # pragma: no cover
            pass

    def from_pretrained(name, **kw):
        calls.append(kw.get("local_files_only", False))
        if not kw.get("local_files_only"):
            raise OSError("You are trying to access a gated repo.") from _err(GatedRepoError)
        return Fake()

    for cls in ("AutoModel", "AutoImageProcessor"):
        monkeypatch.setattr(f"transformers.{cls}.from_pretrained", from_pretrained)
    with caplog.at_level("WARNING", logger="imgad"):
        HFFeatureExtractor("org/gated", device="cpu")._load()
    assert calls == [False, True, False, True]
    assert sum("using the local Hugging Face cache" in r.getMessage() for r in caplog.records) == 1


def test_token_is_passed_to_from_pretrained(monkeypatch):
    seen = {}

    def from_pretrained(name, **kw):
        seen.update(kw)
        raise RuntimeError("stop here")

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", from_pretrained)
    with pytest.raises(RuntimeError, match="stop here"):
        HFFeatureExtractor("org/m", device="cpu", token=SECRET)._load()
    assert seen["token"] == SECRET


# ----------------------------------------------------------------------------- benchmark + CLI
def test_benchmark_checks_access_before_any_work(fake_hub, image_folder, monkeypatch):
    fake_hub.repos = {"org/gated": {"gated": "manual", "access": False}, "org/public": {}}
    extracted = []
    monkeypatch.setattr(HFFeatureExtractor, "extract", lambda self, *a, **k: extracted.append(1))
    ds = data.mvtec_ad(image_folder, "cat")
    with pytest.raises(GatedModelError, match="org/gated") as err:
        benchmark(["org/public", "org/gated", {"type": "stats"}], ["knn"], [ds], verbose=False)
    assert "org/public" not in str(err.value) and not extracted


def test_cli_hf_check(fake_hub, tmp_path, capsys):
    fake_hub.repos = {"org/public": {}, "org/gated": {"gated": "auto", "access": True}}
    fake_hub.users = {SECRET: "alice"}
    tok = tmp_path / "tok"
    tok.write_text(SECRET)
    tok.chmod(0o600)
    main(["hf-check", "org/public", "org/gated", "--hf-token-file", str(tok)])
    out = capsys.readouterr().out
    assert "Token: explicit token (user alice)" in out and "[ok] org/gated" in out and SECRET not in out

    cfg = tmp_path / "b.yaml"
    cfg.write_text("extractors: [org/gated, {type: stats}]\n")
    with pytest.raises(SystemExit) as e:
        main(["hf-check", "--config", str(cfg)])
    assert e.value.code == 1 and "[NO] org/gated" in capsys.readouterr().out


def test_cli_gated_error_is_clean(fake_hub, image_folder, tmp_path, capsys):
    fake_hub.repos = {"org/gated": {"gated": "manual", "access": False}}
    cfg = tmp_path / "b.yaml"
    cfg.write_text(f"data: {{type: mvtec, root: {image_folder}}}\nextractors: [org/gated]\ndetectors: [knn]\n")
    with pytest.raises(SystemExit) as e:
        main(["benchmark", "--config", str(cfg), "-q"])
    err = capsys.readouterr().err
    assert e.value.code == 2 and "error: Some Hugging Face models cannot be accessed" in err
    assert "Traceback" not in err


def test_cli_hf_login(monkeypatch, tmp_path, capsys):
    calls = {}
    monkeypatch.setattr("huggingface_hub.whoami", lambda token=None: {"name": "alice"})
    monkeypatch.setattr("huggingface_hub.login", lambda token, add_to_git_credential: calls.update(token=token))
    tok = tmp_path / "tok"
    tok.write_text(SECRET)
    tok.chmod(0o600)
    main(["hf-login", "--hf-token-file", str(tok)])
    assert calls["token"] == SECRET and "Logged in to Hugging Face as alice" in capsys.readouterr().out
