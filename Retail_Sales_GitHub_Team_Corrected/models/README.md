# Local model files

Model artifacts are **not included** in the public repository. The original `sales_model.pkl` was about 1.52 GB uncompressed, larger than GitHub's normal 100 MB file limit. The feature-list pickle was also omitted so both artifacts have one documented provenance.

You can create the two expected files by running `python src/train_model.py` after preprocessing:

- `models/sales_model.pkl`
- `models/model_features.pkl`

Alternatively, if you already have the original **trusted** model and feature-list files, place both here manually (not via Git). The original model was trained using the 10 extended features listed in the README.

Warning: `joblib.load` uses pickle deserialization and may execute code in an untrusted file; never load unknown third-party model files. Compatibility with a different scikit-learn version is not guaranteed. Training can require substantial time, memory and disk space; no cloud-hosted model is provided.
