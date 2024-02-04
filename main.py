from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from skimage import data, color, util
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import streamlit as st
import cv2
from skimage import color, filters, morphology, measure, segmentation
 
model = load_model('FV.h5')
 
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}
 
calories_dict = {
    'apple': 52, 'banana': 89, 'beetroot': 43, 'bell pepper': 20, 'cabbage': 25, 'capsicum': 40,
    'carrot': 41, 'cauliflower': 25, 'chilli pepper': 40, 'corn': 86, 'cucumber': 15, 'eggplant': 25,
    'garlic': 149, 'ginger': 80, 'grapes': 69, 'jalepeno': 29, 'kiwi': 61, 'lemon': 29, 'lettuce': 5,
    'mango': 60, 'onion': 40, 'orange': 43, 'paprika': 92, 'pear': 57, 'peas': 81, 'pineapple': 50,
    'pomegranate': 83, 'potato': 77, 'raddish': 16, 'soy beans': 173, 'spinach': 23, 'sweetcorn': 86,
    'sweetpotato': 86, 'tomato': 18, 'turnip': 28, 'watermelon': 30
}
 
 
average_weights_dict = {
    'apple': 150, 'banana': 120, 'beetroot': 150, 'bell pepper': 120, 'cabbage': 100, 'capsicum': 120,
    'carrot': 50, 'cauliflower': 200, 'chilli pepper': 50, 'corn': 150, 'cucumber': 300, 'eggplant': 200,
    'garlic': 5, 'ginger': 20, 'grapes': 100, 'jalepeno': 20, 'kiwi': 100, 'lemon': 50, 'lettuce': 100,
    'mango': 300, 'onion': 100, 'orange': 150, 'paprika': 120, 'pear': 150, 'peas': 50, 'pineapple': 120,
    'pomegranate': 200, 'potato': 150, 'raddish': 50, 'soy beans': 100, 'spinach': 100, 'sweetcorn': 150,
    'sweetpotato': 150, 'tomato': 100, 'turnip': 150, 'watermelon': 500
}
 
 
def processed_img(img):
    # Resize the input image to match the expected input shape of the model (224x224 pixels)
    img = img.resize((224, 224))
   
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return labels[predicted_class]
 
 
def fetch_calories(prediction):
    try:
        # Convert prediction to lowercase for case-insensitive matching
        prediction_lower = prediction.lower()
 
        # Check if prediction exists in the calories_dict
        if prediction_lower in calories_dict:
            # Fetch calories from the dictionary
            calories = calories_dict[prediction_lower]
            return f"calories: {calories} grams"   # Convert to string for consistency
        else:
            st.warning("Calories information not available for {}".format(prediction))
            return None
    except Exception as e:
        st.error("Error fetching calories. {}".format(e))
        return None
 
def fetch_weights(prediction):
    try:
        # Convert prediction to lowercase for case-insensitive matching
        prediction_lower = prediction.lower()
 
        # Check if prediction exists in the average_weights_dict
        if prediction_lower in average_weights_dict:
            # Fetch average weight from the dictionary
            weight = average_weights_dict[prediction_lower]
            return f"Average weight: {weight} grams"  # Display more context
        else:
            return f"Average weight information not available for {prediction}"
    except Exception as e:
        return f"Error fetching average weight: {e}"
 
def find_blobs(image):
    # Convert PIL image to NumPy array
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    binary = gray > threshold_otsu(gray)
    binary_cleared = clear_border(closing(binary, square(3)))
    label_image = label(binary_cleared)
    image_label_overlay = label2rgb(label_image, image=image_np)
    regions = regionprops(label_image)
    for region in regions:
        centroid_row, centroid_col = region.centroid
        radius = np.sqrt(region.area) / 2
        cv2.circle(image_np, (int(centroid_col), int(centroid_row)), int(radius), (0, 255, 0), 2)
    st.image(image_np, caption="Labeled Image with Marked Blobs", use_column_width=True)
 
def main():
    st.title("Food Image Classifier and Calories")
 
    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
 
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
 
        # Classify the image
        prediction = processed_img(image)
        if prediction:
            st.subheader(f"Predicted Label: {prediction}")
 
            # Fetch and display calories
            calories_info = fetch_calories(prediction)
            if calories_info:
                st.info(calories_info)
 
            weight_info = fetch_weights(prediction)
            if weight_info:
                st.info(weight_info)
       
        # Find and display blobs
        find_blobs(image)
 
if __name__ == "__main__":
    main()