
import tensorflow as tf


datadir = "../Healthy_n_Unhealthy_4_Training_CM"

# OPTIONAL - Test a random sample image that the model has not seen:
# Read in image
plantImage = tf.io.read_file("../UnhealthyLeaves/UnhealthyLeavesTestPreprocessed/O_Unhealthy_IMG_2008.jpg")

# Turn file into a tensor
plantImage = tf.image.decode_image(plantImage)

# Resize image
plantImage = tf.image.resize(plantImage, size=[256, 256])

# Normalize data
plantImage = plantImage / 255

categories = ['HealthyLeaves', 'UnhealthyLeaves']
print("Loading model")
saved_model = tf.keras.models.load_model("../SavedModels/CNN_Model_3")
prediction = saved_model.predict(tf.expand_dims(plantImage, axis=0))
print(f"tf.round(prediction): {tf.round(prediction)}")

predicted_class = categories[int(tf.round(prediction))]
print(f"The predicted_class is : {predicted_class} with prediction: {prediction}")







# References:
# https://www.youtube.com/watch?v=eMMZpas-zX0
# https://www.youtube.com/watch?v=ad-Qc42Kbx8&ab_channel=DerekBanas
# https://github.com/derekbanas/tensorflow/blob/main/tf_cnn_tut.ipynb