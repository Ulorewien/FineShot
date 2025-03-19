# FineShot

This project aims to leverage VLMs to achieve fine-grained object detection using zero-shot learning.



![Example Image](https://github.com/Ulorewien/FineShot/blob/main/test/imgs/output.jpg?raw=True)

## Download Datasets

To download the images for both the datasets (RefCOCO, Visual Genome) use the links given below:

| Dataset Name | Dataset Link |
| ----------- | ------------ |
| RefCOCO | [Link](https://drive.google.com/file/d/19IlJEOBx071QQ_4J6Y5Dpgs5BwOZSTus/view?usp=sharing) |
| Visual Genome | [Link](https://drive.google.com/file/d/1SjJ25bB8N3n4V5J2BekAYZDJwshMjOaf/view?usp=sharing) |

Place the images under test/refcoco_images and test/vg_images folders respectively.

## Setting up the Environment

1. Clone this repository to your local machine:

```sh
git clone https://github.com/Ulorewien/FineShot.git
cd FineShot
```

2. Create a new Conda environment with Python 3.10 (or the required version).

    ```sh
    conda create -n fineshot python=3.10
    conda activate fineshot
    ```

3. Install the required dependencies from the `requirements.txt` file.

    ```sh
    pip install -r requirements.txt
    ```

Now your environment is set up and you can proceed with running the experiments.

## Running Experiments

To run all our experiments we have used **NVIDIA GeForce RTX 3090** GPU with 24GB memory. For VLM, we used **4-bit quantization** for fitting the model into our memory constraints and for efficient inferencing. We will consider adding support for leveraging multi-gpu setup in the future.

To run the experiments you can directly run the evaluate model python scripts to get the model predicts on the datasets. You will have to change the file paths mentioned in the files to make sure your code run properly.

```sh
python3 evaluate_owlv2.py
python3 evaluate_yolo_vlm.py
.
.
```

Finally you can run the `evaluate_map.py` file to compile the mAP score for each configuration.