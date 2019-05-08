#### Reading data ####
#fit_generator
#flow_images_from_directory
#image_data_generator

img_height <- 200
img_width <- 320
batch_size <- 32
epochs <- 3
channels <- 1
train_image_dir = "Train"
val_image_dir = "Validation"

train_datagen = image_data_generator(rescale = 1/255)
val_datagen = image_data_generator(rescale = 1/255)

train_generator = flow_images_from_directory(
  directory = train_image_dir,
  generator = train_datagen,
  target_size = c(img_width, img_height),
  batch_size = batch_size,
  color_mode = "grayscale",
  class_mode = "categorical",
  shuffle = T
)

val_data_generator = flow_images_from_directory(
  directory = val_image_dir,
  generator = val_datagen,
  target_size = c(img_width, img_height),
  batch_size = batch_size,
  color_mode = "grayscale",
  class_mode = "categorical",
  shuffle = T
)

model <- keras_model_sequential()

model %>%
  
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

  layer_conv_2d(
    filter = 32, kernel_size = c(3,3), padding = "same", 
    input_shape = c(img_width, img_height, channels)
  ) %>%
  layer_activation("relu") %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_flatten() %>%

  # layer_dense(units = 16, activation = "relu") %>%
  
  layer_dense(units = 4, activation = "softmax")
  

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)
# optimizer_adam(lr = 0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0),

model %>% fit_generator(
  train_generator,
  steps_per_epoch = as.integer(47201/batch_size),
  epochs = epochs,
  validation_data = val_datagen,
  validation_steps = as.integer(10195/batch_size),
#  callbacks = callback_early_stopping(monitor = "val_loss", patience = 2),
  verbose = 1,
  workers = 4
)

save_model_hdf5(model, filepath = "model1.hdf")

# model %>%
#   
#   layer_conv_2d(
#     filter = 32, kernel_size = c(3,3), padding = "same", 
#     input_shape = c(img_width, img_height, channels)
#   ) %>%
#   layer_activation("relu") %>%
#   
#   layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
#   layer_activation("relu") %>%
#   
#   layer_max_pooling_2d(pool_size = c(2,2)) %>%
#   layer_dropout(0.25) %>%
#   
#   layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same") %>%
#   layer_activation("relu") %>%
#   layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
#   layer_activation("relu") %>%
#   
#   layer_max_pooling_2d(pool_size = c(2,2)) %>%
#   layer_dropout(0.25) %>%
#   
#   layer_flatten() %>%
#   layer_dense(512) %>%
#   layer_activation("relu") %>%
#   layer_dropout(0.5) %>%
#   
#   layer_dense(4) %>%
#   layer_activation("softmax")
