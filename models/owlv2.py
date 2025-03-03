from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

owlv2_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
owlv2_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")

def owl_v2(image_path, class_labels, confidence_threshold=0.1):
    image = Image.open(image_path)
    inputs = owlv2_processor(
        text=class_labels, 
        images=image, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(owlv2_model.device)
        
    with torch.no_grad():
        outputs = owlv2_model(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]])
    results = owlv2_processor.post_process_object_detection(
        outputs=outputs, 
        target_sizes=target_sizes, 
        threshold=confidence_threshold
    )[0]
    
    return {k: v.cpu().tolist() for k, v in results.items()}


if __name__ == "__main__":
    from .utils import render_image
    class_labels = ["golden dog", "black cat"]
    detections = owl_v2("/data/yashowardhan/FineShot/test/imgs/dogs.jpeg", class_labels)
    print(detections)
    render_image(
        "/data/yashowardhan/FineShot/test/imgs/dogs.jpeg", 
        detections["boxes"], 
        detections["labels"], 
        class_labels, 
        "/data/yashowardhan/FineShot/test/imgs/output.jpg"
    )