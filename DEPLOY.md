# Deploy

The frontend (`streamlit_app.py`) is a pure-CPU Streamlit app — it deploys to Streamlit
Community Cloud for free, straight from GitHub. Two stages: **push to GitHub**, then
**connect Streamlit Cloud**.

## 1. Push to GitHub

`gh` is installed on this machine but not logged in, and login is interactive, so run
these yourself (one time):

```bash
# authenticate (opens a browser)
gh auth login

# create the repo under your account and push everything in one shot
gh repo create yolov11s-int8-orin --public --source=. --remote=origin --push --description \
  "INT8 PTQ of YOLOv11s for Jetson Orin NX — host-testable toolkit + Streamlit demo"
```

No `gh`? Create an empty repo named `yolov11s-int8-orin` on github.com, then:

```bash
git remote add origin https://github.com/<you>/yolov11s-int8-orin.git
git branch -M main
git push -u origin main
```

> The repo is already committed locally (`git log` shows the initial commit). Datasets,
> weights, and engines are git-ignored; the calibration cache is intentionally NOT
> ignored (commit it after your first device run).

## 2. Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with GitHub (authorize it once).
2. **Create app → Deploy a public app from GitHub**, then set:
   - **Repository:** `<you>/yolov11s-int8-orin`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
3. Click **Deploy**. First build installs `requirements.txt` (streamlit + numpy) and
   boots the app; you get a URL like `https://<you>-yolov11s-int8-orin.streamlit.app`.

That's it — no secrets, no config. `requirements.txt` deliberately excludes
torch/tensorrt/cuda so the Cloud build stays small and fast (the app only needs the
numpy host core).

### Notes
- Streamlit Cloud reads **`requirements.txt`** only; `requirements-host.txt` /
  `requirements-device.txt` are for local/device use and are ignored by Cloud.
- The app finds `qint` via `sys.path` (it adds `src/`), so no packaging/install step is
  needed on Cloud.
- To update the live app, just `git push` — Cloud auto-redeploys on every push to `main`.
