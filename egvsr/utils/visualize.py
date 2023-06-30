import cv2
import numpy as np
from absl.logging import info


def visualize_image(image, scale, path):
    image = image.permute(1, 2, 0).cpu().numpy() * 255
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    info(f"Saving image with scale({scale}) to {path}.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename=path, img=image)


def visualize_event(event, scale, path):
    event = event.permute(1, 2, 0).cpu().numpy()
    event_image = np.zeros((event.shape[0], event.shape[1], 3)) + 255
    event_image[event[:, :, 0] > 0] = [0, 0, 255]
    event_image[event[:, :, 1] > 0] = [255, 0, 0]
    event_image = event_image.astype(np.uint8)
    event_image = cv2.resize(event_image, (0, 0), fx=scale, fy=scale)
    info(f"Saving event with scale({scale}) to {path}.")
    cv2.imwrite(filename=path, img=event_image)


def visualize_event_alpx(event, path):
    event = event.squeeze().permute(1, 2, 0).cpu().numpy()
    event_abs = np.abs(event)
    event_abs = np.sum(event_abs, axis=2)
    event = np.sum(event, axis=2)
    event_image = np.zeros((event.shape[0], event.shape[1], 3)) + 255
    event_image[event_abs > 0] = [0, 0, 0]
    event_image[event > 0] = [0, 0, 255]
    event_image[event < 0] = [255, 0, 0]
    event_image = event_image.astype(np.uint8)
    info(f"Saving event to {path}.")
    cv2.imwrite(filename=path, img=event_image)
