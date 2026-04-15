# ML Baseline
## Author: [Łukasz Murza](https://github.com/XEN00000)

[Back to README](../README.md)

This module implements the baseline R3D-18 model for human behavior recognition. It provides a standard entrypoint for fine-tuning on video subsets and validating the resulting checkpoints.

## Training

To run fine-tuning on the baseline model using the configured subset:
```bash
docker compose run --rm inference python -m scripts.train
```

**Training outputs:**
* **Model checkpoints:** Saved to `./data/logs/checkpoints/`.
* **Training metrics:** Performance logs (JSONL) are saved to `./data/logs/metrics/`.

**Class Configuration:**
The `num_classes` parameter is intentionally derived dynamically from the dataset manifest during runtime. This approach ensures that the model architecture (specifically the final linear layer) always matches the number of unique labels present in the current data subset, avoiding configuration drift and manual errors.

## Validation

To evaluate the performance of a specific checkpoint on the validation split, run:
```bash
docker compose run --rm inference python -m scripts.validate --checkpoint data/logs/checkpoints/baseline_epoch_10.pth
```

**Validation outputs:**
* **Summary report:** A JSON file containing accuracy and evaluated classes is saved to `./data/logs/metrics/validation_summary.json`.

### Selected Action Classes (Sprint 2)
The following classes were selected for the first fine-tuned checkpoint:
- `walking`
- `running`
- `falling`