# About

## License

imgad is distributed under the [Apache 2.0 license](https://github.com/joheras/imgad/blob/main/LICENSE).
Models and datasets used with it keep their own licenses: check them before use, especially the
research-only and non-commercial licenses common in medical imaging.

## Acknowledgements

This work is partially supported by Agencia de Desarrollo Económico de La Rioja [ADER 2022-I-IDI-00015].

## Releases

| Version | Highlights |
|---|---|
| 0.1.0 | Feature pipelines, classical detectors, Anomalib backend with anomaly maps, statistical benchmarking, markdown reports |
| 0.2.0 | Gated and private Hugging Face models; benchmark progress reporting |
| 0.3.0 | Parallel benchmarks with several models per GPU and memory checks |

The full history is on [GitHub releases](https://github.com/joheras/imgad/releases).

## Contributing

Issues and pull requests are welcome at [github.com/joheras/imgad](https://github.com/joheras/imgad).
The test suite runs offline:

```bash
pip install -e ".[dev]"
HF_HUB_OFFLINE=1 pytest
```

Documentation lives in `docs/` and is built with MkDocs Material:

```bash
pip install -e ".[docs]"
mkdocs serve
```
