
import tensorflow as tf



# OPTIONAL - Test a random sample image that the model has not seen:
# Read in image
plantImage = tf.io.read_file("../HealthyLeaves/HealthyLeavesTrainPreprocessed/O_Healthy_20210208_101726.jpg")
# Turn file into a tensor
plantImage = tf.image.decode_image(plantImage)

# Resize image
plantImage = tf.image.resize(plantImage, size=[256, 256])

# Normalize data
plantImage = plantImage / 255

categories = ['HealthyLeaves', 'UnhealthyLeaves']
print("Loading model")
saved_model = tf.keras.models.load_model("../SavedModels/CNN_Model_1")
prediction = saved_model.predict(tf.expand_dims(plantImage, axis=0))
print(f"tf.round(prediction): {tf.round(prediction)}")

predicted_class = categories[int(tf.round(prediction))]
print(f"The predicted_class is : {predicted_class} with prediction: {prediction}")




# References:
# https://www.youtube.com/watch?v=ad-Qc42Kbx8&ab_channel=DerekBanas
