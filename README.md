# solar_segmentation

This project demonstrates how to use a U-Net model with TensorFlow and Keras to detect solar panels in satellite images. 

The U-Net model consists of an encoder and a decoder. The encoder is based on the VGG16 model with pre-trained weights on Imagenet, and the decoder is trained on a custom dataset of satellite images with solar panel annotations.

The steps to create the U-Net model are described in the `panel_segmentation` notebook and The `test_model_notebook` notebook shows how to use the trained model to make predictions on new images.
