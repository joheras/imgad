# Gated Hugging Face models

Some models (e.g. DINOv3) are **gated**: accept their conditions on the model page and authenticate
with an [access token](https://huggingface.co/settings/tokens). imgad uses, in this order: an explicit
token (`token=` / `--hf-token-file`), the `HF_TOKEN` environment variable, or the token stored by
`imgad hf-login` (same as `hf auth login`). Tokens are **never** saved in `.joblib` files, configs,
feature-cache keys, reports or logs.

```bash
imgad hf-login                                            # paste the token (input hidden)
imgad hf-check facebook/dinov3-vitb16-pretrain-lvd1689m   # access status and what to do
imgad hf-check --config configs/mvtec_benchmark.yaml      # every Hub model of a benchmark
imgad benchmark --config cfg.yaml --hf-token-file ~/.hf_token   # or fit/predict/calibrate
```

```python
HFFeatureExtractor("facebook/dinov3-vitb16-pretrain-lvd1689m", token=my_token)
AnomalyPredictor.load("model.joblib", token=my_token)      # the token is not stored in the file
benchmark(..., token=my_token)                            # checks access to every Hub model first
```

- A benchmark checks access to all its Hub models **before** doing any work and lists, for each model
  that cannot be loaded, what to do (accept the conditions, log in, replace a rejected token…).
- If you have no access but the model is already in the local Hugging Face cache, it is loaded from the
  cache (with a warning).
- There is deliberately no `--hf-token VALUE` option: a token on the command line ends up in the shell
  history and the process list.
