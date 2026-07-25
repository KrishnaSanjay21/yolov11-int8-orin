# data/ (not committed)

Place these on the device (git-ignored — datasets don't belong in the repo):

```
data/
  calib/            # calibration images (JP/PNG); 512+ recommended so the 32/128/512
                    #   sweep all draw from the same pool. Must match the deployment
                    #   distribution and class mix.
  val/              # validation images for accuracy eval
  val.json          # COCO-format instances annotations for val/ (see qint.coco)
```

`scripts/04_calibrate_int8.py --images data/calib` and `scripts/07_run_accuracy.py
--img-dir data/val --ann data/val.json` read from here.

If you use COCO val2017, `val.json` is the stock `instances_val2017.json`. For a custom
set, any COCO instances dict with `images`, `annotations` (xywh bbox), and `categories`
works — `qint.coco.load_coco_gt` handles the dense remap.
