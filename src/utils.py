
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import numpy as np
from anomalib.utils.post_processing import superimpose_anomaly_map







#PLOTTING STUFF
def show_good_samples(training_datamodule):
    number_of_images_to_show = 3
    _, data = next(enumerate(training_datamodule.train_dataloader()))
    show_images_with_labels(data, num_images=number_of_images_to_show)

def show_anomaly_samples(training_datamodule):
    number_of_images_to_show = 3
    _, data = next(enumerate(training_datamodule.test_dataloader()))

    show_images_with_labels(data, num_images=number_of_images_to_show)

def show_images_with_labels(sample, num_images):
    # Set up the figure with 3 columns and 1 row
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for i in range(num_images):
        # Load the image and resize it
        image = Image.open(sample["image_path"][i])
        image.thumbnail((256, 256), Image.Resampling.LANCZOS)

        # Determine the label (good image or anomaly)
        label = 'Good Image' if sample["label"][i].numpy() == 0 else 'Anomaly'

        # Display the image on the corresponding axis
        axes[i].imshow(image)
        axes[i].set_title(label)
        axes[i].axis('off')  # Hide the axes for a cleaner look

    plt.show()


#DISPLAYING STUFF
def display_anomalymap(image, anomaly_map, score, ax, save_path, threshold=0.85):
    # Process the anomaly map
    anomaly_map = anomaly_map.cpu().numpy()
    anomaly_map = cv2.resize(255 * (1 - anomaly_map[0]), (image.shape[1], image.shape[0]))
    anomaly_map = np.array(anomaly_map).astype(int)

    # Superimpose the anomaly map onto the original image
    heat_map = superimpose_anomaly_map(anomaly_map=anomaly_map, image=image, normalize=False)

    if score > threshold:
        image_title = f"Anomaly!  S={score:.3f}"
    else:
        image_title = f"No Anomaly!  S={score:.3f}"

    # Plotting the image and anomaly map
    ax.imshow(heat_map)
    ax.set_title(image_title)
    ax.axis('off')  # Hide the axis for a cleaner look

    # Save the image instead of showing it
    plt.savefig(save_path,
                bbox_inches='tight', pad_inches=0.1)  # Save the figure with tight bounding box

    print(f"Image saved as {save_path}")


#SAVING STUFF
def save_images(predictions):
    display_imgs = []
    # Assuming `predictions` is a list of prediction dictionaries
    for i, prediction in enumerate(predictions):
        prediction = predictions[i]
        image = cv2.imread(prediction['image_path'][0])  # Original input image
        anomaly_map = prediction['anomaly_maps'][0]  # Replace with your anomaly map
        score = prediction['pred_scores'][0]  # Replace with your anomaly score
        save_path = 'custom_results/output_image' + str(i) + '.png'  # Path where the image will be saved

        # Create the figure and axes for displaying the image
        fig, ax = plt.subplots()
        display_anomalymap(image, anomaly_map, score, ax, save_path)

