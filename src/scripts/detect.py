from torchvision import transforms
from tqdm import tqdm
from src.utils import *
from PIL import Image, ImageDraw, ImageFont

from src.utils import rev_label_map, label_color_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = '../../weights/checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform

    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)

    font = ImageFont.truetype("./arial.ttf", original_image.height // 50)

    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        left, top, right, bottom = det_boxes[i].tolist()
        label_color = label_color_map[det_labels[i]]
        draw.rectangle(((left, top), (right, bottom)), outline=label_color, width=4)

        text_location = (left + 5, top - 25)
        label_text = det_labels[i].upper()
        text = '{0} {1:.2f}'.format(label_text, det_scores[0][i])

        left, top, right, bottom = draw.textbbox(text_location, text, font=font)
        draw.rectangle((left - 5, top - 5, right + 5, bottom + 5), fill=label_color)
        draw.text(text_location, text, font=font, fill="white")

    del draw

    return annotated_image


def process_images(input_folder, output_folder, min_score=0.6, max_overlap=0.2, top_k=200):
    os.makedirs(output_folder, exist_ok=True)
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            original_image = Image.open(img_path).convert('RGB')
            annotated_image = detect(original_image, min_score, max_overlap, top_k)
            output_path = os.path.join(output_folder, filename)
            annotated_image.save(output_path)


if __name__ == '__main__':
    input_folder = '../../data/MVTec/leather/train/'
    output_folder = '../../data/MVTec/leather/results/'
    process_images(input_folder, output_folder)
