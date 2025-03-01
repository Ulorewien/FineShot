# Visual Genome Subset

This data contains a subset of the Visual Genome dataset, specifically focusing on region descriptions. The data is saved in a JSON file format and processed for easy consumption by machine learning models. The subset consists of information extracted from the first 1000 images from the Visual Genome dataset, including the image ID and region descriptions with coordinates.

## Dataset Structure

The dataset is stored in the `vg_subset.json` file, which is structured as follows:

```json
[
  {
    "image_id": <image_id>,
    "regions": [
      {
        "coordinates": [x1, y1, x2, y2],
        "phrase": <description>
      },
      ...
    ]
  },
  ...
]

