Run the following commands in PowerShell from the `raspberrypi-smart-cart-ai-vision-module` directory to generate the processed images.

```powershell
conda activate smartcart-ai

python tools\process_sample_images.py `  --input-dir demos\sample_image_processing\original_images`
  --output-dir demos\sample_image_processing\processed_images `  --device cpu`
  --draw-space processed `
  --filter-mode autobash
```
