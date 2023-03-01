# Eye-Shield

### Background
Eye-Shield is a mobile device screen privacy system. The algorithm alters images to make them appear blurry or pixelated at a distance while retaining information up close.

### Installation
Using python version 3.9.7
```
conda env create -f dependencies/environment_{OSNAME}.yml
Or
pip install -r dependencies/requirements.txt
pip install -e .
```

Additional requirements for Nvidia GPUs with CUDA:
```
pip install cupy
```

Additional requirements for protecting videos:
```
pip install ffmpeg-python
```

### Basic Usage

Call protect_image() in `eyeshield.py`. It takes a numpy image and user-specified arguments such as strength and resolution size to determine the optimal parameters. You can install it with `pip install eyeshield` and use it like the following:
```
from eyeshield
```

### Development
```
./scripts/run_hideimage.sh
```
Runs the Eye-Shield algorithm with certain parameters on high-res image and mobile UI datasets.

```
./scripts/run_computescores.sh
```
Calculates SSIM and entropy by comparing original images, target images, and protected images.

```
./scripts/run_hidevideo.sh
```
Runs the Eye-Shield algorithm with certain parameters for video files.

```
./scripts/run_hideimage_gpu.sh
```
Runs the with-GPU Eye-Shield algorithm.

```
./scripts/run_google_cloud.sh
```
Runs the evaluation on Google Cloud Vision. Requires additional dependencies and API access keys.


```
./scripts/run_plots.sh
```
Runs scripts to process and plot the experiment results.

### Licensing

Eye-Shield is available under dual licensing. For commercial purposes, we recommend discussing licensing with our university patent office. For all other purposes, please use Eye-Shield under the GPL open-source license found in LICENSE.