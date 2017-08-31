import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests



# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    # Names
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Load model
    tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)
    
    # Get tensors
    image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)


    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output


    FCN structure
    =========


    -----vgg_layer3             (20 x 27 x 256 )
    | ---vgg_layer4             (10 x 36 x 512 )
    | |  vgg_layer7             ( 5 x 18 x 4096)
    | |
    | |  (above is encoder, frozon from VGG)
    | |  (below is decoder, learned during training)
    | |
    | |  vgg_layer7_resampled   ( 5 x 18 x 2)
    | |  decoder_layer1         (10 x 36 x 2)
    | ---(vgg_layer4 resampled) (10 x 36 x 2)
    |    combined_layer1        (10 x 36 x 2)
    |    decoder_layer2         (20 x 72 x 2)
    -----(vgg_layer3 resampled) (20 x 72 x 2)
         combined_layer2        (20 x 72 x 2)
         final_layer           (160 x 576 x 2)
    
    The decoder layers basically perform transposed convolutions
    and implement skip layer (resampling where necessary to get
    consistent kernel sizes) 
    """


    # Start by freezing VGG
    # Since we are not re-training the encoder part
    vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)
    vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
    vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)


    # 1x1 convolution on vgg_layer7_out to reduce to num_classes kernels
    # in:   ?x5x18x4096
    # out:  ?x5x18x2
    vgg_layer7_out_resampled = tf.layers.conv2d(vgg_layer7_out,num_classes,1,strides=(1,1),kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))

    # Upsample vgg_layer7_out_resampled by factor of 2 
    # In:   ?x5x18x2 
    # Out:  ?x10x36x2
    decoder_layer1 = tf.layers.conv2d_transpose(vgg_layer7_out_resampled,num_classes,kernel_size=(2,2),strides=(2,2),padding='valid',kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))

    # Resample vgg_layer4_out to reduce to num_classes kernels
    # In:  ?x10x36x512 
    # OUt: ?x10x36x2
    vgg_layer4_out_resampled = tf.layers.conv2d(vgg_layer4_out,num_classes,1,strides=(1,1),kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))

    # Combine to complete skip layer
    combined_layer1 = tf.add(decoder_layer1, vgg_layer4_out_resampled)

    # Upsample combined_layer1 by factor of 2 
    # In:  ?x10x36x2 
    # Out: ?x20x72x2
    decoder_layer2 = tf.layers.conv2d_transpose(combined_layer1,num_classes,kernel_size=(2,2),strides=(2,2),padding='valid',kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))

    # Sesample vgg_layer3_out to reduce to num_classes kernels
    # In:  ?x20x72x256 
    # Out: ?x20x72x2
    vgg_layer3_out_resampled = tf.layers.conv2d(vgg_layer3_out,num_classes,1,strides=(1,1),kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))

    # Combine to complete skip layer
    combined_layer2 = tf.add(vgg_layer3_out_resampled, decoder_layer2)

    # Upsample combined_layer2 by factor of 8 
    # In:  ?x20x72x2 
    # Out: ?x160x576x2
    final_layer = tf.layers.conv2d_transpose(combined_layer2,num_classes,kernel_size=(8,8),strides=(8,8),padding='valid',kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))

   
    return final_layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # Reshape logits and labels to get mean cross entropy
    logits = tf.reshape(nn_last_layer,(-1,num_classes))
    labels_vec = tf.reshape(correct_label,(-1,num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_vec))

    # Use Adam Optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)

    return logits,train_op,cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """


    # Iterate through epochs
    for epoch in range(epochs):
        batch_num = 0
        # Iterate through batches
        for batch_x,batch_y in get_batches_fn(batch_size):
    
            # Run actual training with loss
            _,cost = sess.run([train_op,cross_entropy_loss],feed_dict={input_image:batch_x,correct_label:batch_y,learning_rate:10**-3, keep_prob:0.5})

            # Report progress
            print('')
            print('Epoch: %i, batch number: %i, cost: %.3f'%(epoch,batch_num,cost))
            batch_num += 1

tests.test_train_nn(train_nn)


def run_nn():

    # Epochs and batch_size chosen through experiments 
    # and reviewer's recommendations
    epochs = 18
    batch_size = 20

    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Placeholders
    correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
    learning_rate = tf.placeholder(tf.float32)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        print('')
        print('Start loading VGG')

        # load_vgg
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        print('')
        print('Done loading VGG')


        # layers
        final_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # optimize
        logits,train_op,cross_entropy_loss = optimize(final_layer, correct_label, learning_rate, num_classes)

        # Initialize
        sess.run(tf.global_variables_initializer())

        # Train
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,correct_label, keep_prob, learning_rate)

        # Save
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run_nn()
