# FastVLM Remote Zoo Models for FiftyOne

![FastVLM Demo](./fastvlm-hq.gif)


Apple's FastVLM vision-language models integrated as FiftyOne remote zoo models, enabling seamless visual question answering in your computer vision workflows.

## Available Models

- **FastVLM-0.5B**: Lightweight model suitable for basic VQA tasks
- **FastVLM-1.5B**: Medium-sized model balancing performance and resource usage
- **FastVLM-7B**: Large model offering state-of-the-art performance

## Repository Structure

```text
fastvlm-zoo/
├── __init__.py        # Package initialization with download/load functions
├── zoo.py             # Main model implementation
├── manifest.json      # Model metadata and requirements
└── README.md          # This file
```

## Features

- **Visual Question Answering (VQA)**: Ask natural language questions about images
- **Flexible Prompting**: Customize system and user prompts for specific domains
- **Field-based Prompts**: Use prompts from dataset fields for dynamic questioning
- **Batch Processing**: Efficiently process entire FiftyOne datasets
- **Multi-device Support**: Runs on CUDA, Apple Silicon (MPS), or CPU

## Installation

```python
import fiftyone.zoo as foz

# Register the model source
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/fast_vlm",
    overwrite=True
)

# Download the desired model variant (first time only)
# Choose from: "apple/FastVLM-0.5B", "apple/FastVLM-1.5B", or "apple/FastVLM-7B"
foz.download_zoo_model(
    "https://github.com/harpreetsahota204/fast_vlm",
    model_name="apple/FastVLM-7B"  # Change to desired model variant
```

## Usage Examples

### Complete Example with HuggingFace Dataset

```python
import fiftyone as fo
import fiftyone.utils.huggingface as fouh
import fiftyone.zoo as foz

# Load a dataset from HuggingFace
dataset = fouh.load_from_hub(
    "Voxel51/MashUpVQA",
    max_samples=10,  # Limit samples for testing
    overwrite=True
)

# Register and download the model (first time only)
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/fast_vlm",
    overwrite=True
)
foz.download_zoo_model(
    "https://github.com/harpreetsahota204/fast_vlm",
    model_name="apple/FastVLM-1.5B"  # Choose model variant
)

# Load the model
model = foz.load_zoo_model("apple/FastVLM-1.5B")

# Answer questions from the dataset
dataset.apply_model(model, prompt_field="question")

# Generate creative content with a custom prompt
model.prompt = "Write a lovely poem about what you see here"
dataset.apply_model(model, label_field="poem")

# View results
session = fo.launch_app(dataset)
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | "What is in this image?" | Question/prompt for all images |
| `prompt_field` | str | None | Field containing per-image prompts |
| `label_field` | str | None | Field to store model outputs |
| `system_prompt` | str | None | Custom system prompt |
| `temperature` | float | 0.7 | Generation temperature (0.1-2.0) |
| `max_new_tokens` | int | 512 | Maximum tokens to generate |
| `top_p` | float | 0.90 | Top-p sampling parameter |
| `top_k` | int | 50 | Top-k sampling parameter |
| `device` | str | auto | Device to use ("cuda", "mps", or "cpu") |
| `max_samples` | int | None | Maximum samples to process |

## Requirements

- Python
- PyTorch
- Transformers
- FiftyOne
- CUDA-capable GPU (recommended)
- GPU Memory Requirements:
  - FastVLM-0.5B: 4GB+ GPU memory
  - FastVLM-1.5B: 8GB+ GPU memory
  - FastVLM-7B: 16GB+ GPU memory

## Performance Tips

1. **Use GPU**: The model runs significantly faster on CUDA devices
2. **Batch Processing**: The model processes images individually; consider parallelization for large datasets
3. **Adjust Max Tokens**: Lower `max_new_tokens` for faster inference when detailed responses aren't needed
4. **Device Selection**: The model automatically selects the best available device (CUDA > MPS > CPU)

## Example Notebooks

See the `examples/` directory for Jupyter notebooks demonstrating:
- Basic VQA workflows
- Custom prompt engineering for specific domains
- Field-based dynamic prompting
- Integration with FiftyOne Brain for similarity search
- Multi-modal dataset analysis

## Citation

```bibtex
@InProceedings{fastvlm2025,
  author = {Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari},
  title = {FastVLM: Efficient Vision Encoding for Vision Language Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2025},
}
```

## License

This integration is provided under Apache-2.0 License. The FastVLM model itself is subject to [Apple's licensing terms](https://github.com/apple/ml-fastvlm/blob/main/LICENSE).
