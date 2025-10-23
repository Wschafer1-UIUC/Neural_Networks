###############################################################################
# Filename: supportFuncs
#
# Description: This script hosts support functions for the digit identification
#              script.
#
# Functions
#   - drawAndPredictDigit()
#
###############################################################################
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# function to create a temporary canvas for digit estimation
def drawAndPredictDigit(model_path, canvas_size=28, brush=2):
    ############################################################################
    # Function: drawAndPredictDigit
    #
    # Description: Opens a white 2D canvas for drawing a handwritten digit (0–9)
    #              in black using the mouse. The window shows a large canvas for 
    #              smooth drawing and downsamples the result to 28x28 to match 
    #              MNIST before inference. Press Enter to submit (and run the 
    #              model prediction), 'c' to clear the canvas, or Esc to 
    #              cancel/close without prediction.
    #
    # Inputs:
    #   model_path:   path to a saved Keras model ('.keras' or '.h5')
    #   canvas_size:  side length (pixels) of the square drawing canvas
    #   brush:        brush thickness in pixels for drawing strokes
    #
    # Outputs:
    #   pred_label:   predicted class label or None if cancelled
    #   probs:        model softmax probabilities for classes 0–9
    #   img28:        preprocessed 28x28 grayscale input sent to model
    #
    ############################################################################
    model = tf.keras.models.load_model(model_path)

    win = 'Draw 0-9 (Enter=Submit, c=Clear, Esc=Quit)'
    canvas = np.full((canvas_size, canvas_size), 255, np.uint8)
    drawing, last_pt = False, None

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, last_pt
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing, last_pt = True, (x, y)
            cv2.circle(canvas, (x, y), brush // 2, 0, -1)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.line(canvas, last_pt, (x, y), 0, brush, cv2.LINE_AA)
            last_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing, last_pt = False, None

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 400, 400)
    cv2.setMouseCallback(win, on_mouse)

    pred_label = probs = img28 = None
    while True:
        cv2.imshow(win, canvas)
        k = cv2.waitKey(1) & 0xFFFF
        if k == 13:
            small = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
            small = 255 - small
            img28 = small.astype(np.float32)
            img28 = tf.keras.utils.normalize(img28, axis=1)
            img28 = img28.reshape(1, 28, 28)
            probs = model.predict(img28, verbose=0)[0]
            pred_label = int(np.argmax(probs))
            print(f"\nThis digit is probably a {pred_label}\n")
            vis = img28.reshape(28, 28)
            plt.figure()
            plt.title("Model input (28x28, normalized)")
            plt.imshow(vis, cmap="gray")
            plt.axis("off")
            plt.show()
            break
        elif k == 27:
            break
        elif k in (ord('c'), ord('C')):
            canvas[:] = 255

    cv2.destroyAllWindows()
    return pred_label, probs, img28
