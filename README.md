# ARKitScenes data preparation

Run these commands to download 1,000 samples into `data/raw` and process them into `data/processed_data`.

```bash
python3 scripts/download_data.py --count 1000
python3 scripts/process_data.py
```