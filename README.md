# FineShot

Zero-shot object detection (ZSOD) enables object recognition beyond predefined categories by leveraging vision-language models (VLMs) to associate objects with rich textual descriptions. In this project, we evaluate off-the-shelf ZSOD models, including Owl-V2, YOLO-World, and Qwen2.5-VL, by computing their mean average precision (mAP). While these models demonstrate strong zero-shot capabilities, they require high-quality training data and complex training setups. To explore training-free approaches for fine-grained object detection, we propose three alternative workflows.

First, we use YOLO to extract object crops from images and pass them to Qwen2.5-VL along with user-defined labels, prompting it to determine label matches. Second, we leverage CLIP by extracting object crops with YOLO and computing cosine similarity between CLIP embeddings of crops and user labels. A thresholding mechanism filters low-confidence matches. Third, we combine both approaches by using Qwen2.5-VL to generate textual descriptions for object crops, which are then embedded using CLIP and matched to user labels.

We evaluate our approaches on the RefCOCO and Visual Genome datasets, processing 1,000 images from each to extract bounding boxes and textual descriptions while minimizing computational overhead. Our findings demonstrate that integrating vision-language reasoning with traditional object detection improves fine-grained object recognition without requiring additional training or fine-tuning.

![Example Image](https://github.com/Ulorewien/FineShot/blob/main/test/imgs/output.jpg?raw=True)

## Proposed Workflows
1. YOLO + VLM:
    In this approach, we use YOLO as a base object detector to extract object crops from images. These crops are then passed to Qwen2.5-VL, along with user-defined labels. We prompt Qwen2.5-VL to determine whether a given label matches the object in the crop.

    ![Example Image](https://github.com/Ulorewien/FineShot/blob/main/test/imgs/YOLO_QWEN.jpg?raw=True)

2. YOLO + CLIP:
    This approach replaces Qwen2.5-VL with CLIP, a vision-language model that computes semantic embeddings for both images and text. We extract object crops using YOLO and compute their embeddings using CLIP. Similarly, user-provided labels are converted into text embeddings, and we compute cosine similarity between image and text embeddings.
    ![Example Image](https://github.com/Ulorewien/FineShot/blob/main/test/imgs/YOLO_CLIP.jpg?raw=True)

3. YOLO + VLM + CLIP:
    To further improve detection accuracy, we combined Qwen2.5-VL’s object description capabilities with CLIP’s similarity-based matching. Instead of directly matching labels, we first generate textual descriptions of object crops using Qwen2.5-VL and then use CLIP to compare these descriptions with user labels.
    ![Example Image](https://github.com/Ulorewien/FineShot/blob/main/test/imgs/YOLO_QWEN_CLIP.jpg?raw=True)

## Download Datasets

To download the images for both the datasets (RefCOCO, Visual Genome) use the links given below:

| Dataset Name  | Dataset Link                                                                               |
| ------------- | ------------------------------------------------------------------------------------------ |
| RefCOCO       | [Link](https://drive.google.com/file/d/19IlJEOBx071QQ_4J6Y5Dpgs5BwOZSTus/view?usp=sharing) |
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
