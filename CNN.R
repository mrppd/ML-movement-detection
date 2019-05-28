# Variables
img_height <- 200
img_width <- 320
batch_size <- 64
epochs <- 3
channels <- 1
train_image_dir = "Train"
test_image_dir = "Test"

# Image data generator for the train and validation dataset
train_datagen = image_data_generator(rescale = 1/255,
                                     validation_split = 0.2,
                                     data_format = "channels_last")

# Image data generator for the test images
test_datagen = image_data_generator(rescale = 1/255,
                                    data_format = "channels_last")

# Train generator to import the pictures from the directory, also splits it for train/validation
train_generator = flow_images_from_directory(
  directory = train_image_dir,
  generator = train_datagen,
  target_size = c(img_height, img_width),
  batch_size = batch_size,
  color_mode = "grayscale",
  class_mode = "categorical",
  shuffle = T,
  subset = "training"
)

# Reading the validation images from the directory
val_data_generator = flow_images_from_directory(
  directory = train_image_dir,
  generator = train_datagen,
  target_size = c(img_height, img_width),
  batch_size = batch_size,
  color_mode = "grayscale",
  class_mode = "categorical",
  shuffle = T,
  subset = "validation"
)

# Reading the test images from the directory
test_generator = flow_images_from_directory(
  directory = test_image_dir,
  generator = test_datagen,
  target_size = c(img_height, img_width),
  batch_size = batch_size,
  color_mode = "grayscale",
  class_mode = "categorical",
  shuffle = T
)

# Building the model
model <- keras_model_sequential()

# Using pipe operator to split to define the layers
model %>%
  
  layer_conv_2d(
    filter = 32, kernel_size = c(3,3), padding = "same", 
    input_shape = c(img_height, img_width, channels)
  ) %>%
  layer_activation("relu") %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%

  layer_conv_2d(
    filter = 32, kernel_size = c(3,3), padding = "same", 
    input_shape = c(img_width, img_height, channels)
  ) %>%
  layer_activation("relu") %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%

  layer_conv_2d(
    filter = 32, kernel_size = c(3,3), padding = "same", 
    input_shape = c(img_width, img_height, channels)
  ) %>%
  layer_activation("relu") %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_flatten() %>%

  # layer_dense(units = 16, activation = "relu") %>%
  
  layer_dense(units = 5, activation = "softmax")

# Compiling the model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

# Fiting the model
model %>% fit_generator(
  train_generator,
  steps_per_epoch = as.integer(43496/batch_size),
  epochs = epochs,
  validation_data = val_data_generator,
  validation_steps = as.integer(10874/batch_size),
#  callbacks = callback_early_stopping(monitor = "val_loss", patience = 2),
  verbose = 1,
  workers = 4 # in case of good cpu and gpu we can use multiple threads
)

# Saving and loading functions for the model
save_model_hdf5(model, filepath = "model2.hdf")
model = load_model_hdf5("model2.hdf")

# Evaluating the model on the test dataset, again using a generator since we have big dataset
model %>% evaluate_generator(
  test_generator,
  steps = as.integer(13805/batch_size),
  workers = 4
)