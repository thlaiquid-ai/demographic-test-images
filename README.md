# Demographic Test Images

Balanced demographic test images from the FairFace dataset for evaluating age and gender inference models.

## Dataset Overview

| Attribute | Value |
|-----------|-------|
| Total Images | 540 |
| Age Groups | 9 (0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+) |
| Gender | 2 (Male, Female) |
| Samples per Group | 30 |
| Image Resolution | 224x224 px |
| Format | JPEG |

## File Structure

```
fairface/
├── fairface_0001.jpg
├── fairface_0002.jpg
├── ... (540 images)
└── ground_truth.json
```

## Ground Truth Format

`fairface/ground_truth.json` contains metadata for each image:

```json
[
  {"id": "fairface_0001", "filename": "fairface_0001.jpg", "age": "50-59", "gender": "Male"},
  {"id": "fairface_0002", "filename": "fairface_0002.jpg", "age": "30-39", "gender": "Female"},
  ...
]
```

## Raw URL Access

All files can be accessed via GitHub raw URLs:

**Ground Truth JSON (demographic labels):**
```
https://raw.githubusercontent.com/thlaiquid-ai/demographic-test-images/main/fairface/ground_truth.json
```

**Images:**
```
https://raw.githubusercontent.com/thlaiquid-ai/demographic-test-images/main/fairface/fairface_0001.jpg
```

### Example: Fetching Ground Truth in Python

```python
import requests

# Fetch ground truth data
url = "https://raw.githubusercontent.com/thlaiquid-ai/demographic-test-images/main/fairface/ground_truth.json"
response = requests.get(url)
ground_truth = response.json()

# Each entry contains: id, filename, age, gender
for item in ground_truth[:3]:
    print(f"{item['filename']}: age={item['age']}, gender={item['gender']}")
    image_url = f"https://raw.githubusercontent.com/thlaiquid-ai/demographic-test-images/main/fairface/{item['filename']}"
    print(f"  Image URL: {image_url}")
```

## Data Source

- **Dataset**: [HuggingFaceM4/FairFace](https://huggingface.co/datasets/HuggingFaceM4/FairFace)
- **Original Paper**: Karkkainen, K., & Joo, J. (2021). FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation. WACV 2021.
- **License**: CC BY 4.0

## Usage

This dataset is intended for testing and evaluating demographic inference models, specifically for:
- Age estimation accuracy across different age groups
- Gender classification accuracy
- Bias measurement and mitigation

## License

CC BY 4.0 - See [LICENSE](LICENSE) for details.
