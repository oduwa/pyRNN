import tensorflow as tf
import numpy as np
import nltk
import pickle
import random
import Helper
import argparse

# parse cli arguments
ap = argparse.ArgumentParser(description="Tensorflow RNN for text generation")
ap.add_argument('-t', '--train', help = 'Set this flag to train the RNN from scratch', action='store_true', default=False)
ap.add_argument('-t2', '--train2', help = 'Set this flag to train the RNN from the last point', action='store_true', default=False)
ap.add_argument('-g', '--gen', help = 'Set this flag to generate text', action='store_true', default=False)
ap.add_argument('-f', '--file', help = 'The seed file for training or text generation', required=False, default="fanfic.txt")
ap.add_argument('-m', '--model', help = 'The path for the model to be saved/restored from', required=False, default="tmp/rnn.ckpt")
ap.add_argument('-i', '--iters', help = 'Number of training iterations', required=False, default=10000, type=int)
ap.add_argument('-w', '--words', help = 'Number of words to generate with the model', required=False, default=100, type=int)
args = vars(ap.parse_args())
filename = args['file']
model_path = args['model']
isTrainingPhase = args['train']
isContinuingTraining = args['train2']
isTextGen = args['gen']

if(not isTrainingPhase and not isContinuingTraining and not isTextGen):
    isTrainingPhase = True

# data I/O
data = open(filename, 'r').read()
data_words = Helper.tokenize(data.decode('utf8'))
# Make input into a set to remove duplicates and then make it into a list
words = list(set(data_words))
data_size, vocab_size = len(data_words), len(words)
print 'Data has %d words, %d unique.' % (data_size,vocab_size)
# Create dictionaries mapping a word to an index and vice versa
word_to_ix = { ch:i for i,ch in enumerate(words) }
ix_to_word = { i:ch for i,ch in enumerate(words) }

n_inputs = vocab_size # size of vector input at each timestep
n_steps = 3 # Number of timesteps (ie along the x axis)
n_hidden = 512
batch_size = n_steps*25
learning_rate = 0.001
training_iters = args['iters']#25000#100000

# tf Graph input. None means that the first dimension can be of any size so it represents the batch size
x = tf.placeholder("float", [None, n_steps, n_inputs])
y = tf.placeholder("float", [None, vocab_size])

# Initialise weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

# p is the index into our training data for where we are now.
p = 0
def next_batch():
    global p

    batch_x = np.zeros((batch_size, n_steps, n_inputs))
    batch_y = np.zeros((batch_size, n_inputs))

    for i in range(0,batch_size):
        unit_batch = np.zeros((n_steps, n_inputs))
        batch_idx = 0

        # Wrap around if reached end of training data
        if p+n_steps >= len(data_words):
            p = 0 # go from start of data

        for j in xrange(p, p+n_steps):
            word_vector = np.zeros([vocab_size], dtype=float) # one-hot vector
            word_idx = word_to_ix[data_words[j]]
            word_vector[word_idx] = 1
            unit_batch[batch_idx] = word_vector
            batch_idx+=1
        batch_x[i] = unit_batch
        target = np.zeros([vocab_size], dtype=float)
        target_idx = word_to_ix[data_words[p+n_steps]]
        target[target_idx] = 1
        batch_y[i] = target
        p+=1
    return (batch_x, batch_y)




def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # 2-layer LSTM, each layer has n_hidden units.
    #rnn_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_hidden), tf.contrib.rnn.BasicLSTMCell(n_hidden)])

    # generate prediction
    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']



pred = RNN(x, weights, biases)

# Get class probabilities by applying softmax function
pred_probs = tf.nn.softmax(pred)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()


if(isTrainingPhase):
    print("-------- TRAINING --------")
    # Launch the graph
    with tf.Session() as sesh:
        sesh.run(init)
        step = 1
        try:
            # Keep training until reach max iterations
            while step * (batch_size/n_steps) < training_iters:
                batch_x, batch_y = next_batch()

                # Run optimization op (backprop)
                sesh.run(optimizer, feed_dict={x: batch_x, y: batch_y})

                # print to the terminal a sample every 100 training steps so we can see how its doing
                if step % 100 == 0:
                    # Calculate batch accuracy
                    acc = sesh.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                    # Calculate batch loss
                    loss = sesh.run(cost, feed_dict={x: batch_x, y: batch_y})
                    print "Iter " + str(step*(batch_size/n_steps)) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc)
                step += 1
            print "Optimization Finished!"
        except KeyboardInterrupt:
            print "Training Interrupted"

        # Save model weights to disk
        save_path = saver.save(sesh, model_path)
        print("Model saved in file: %s" % save_path)

        # Generate text with saved model
        # Randomly enerate seed text of 3 words from training data
        rdm_idx1 = word_to_ix[random.choice(data_words)]
        rdm_idx2 = word_to_ix[random.choice(data_words)]
        rdm_idx3 = word_to_ix[random.choice(data_words)]
        input_indices = [rdm_idx1,rdm_idx2,rdm_idx3]
        print("SEED TEXT:")
        txt = ' '.join(ix_to_word[ix] for ix in input_indices)
        print '%s' % (txt, )

        # For each word to be generated
        n_gen_words = args['words']
        written_words = []
        for i in range(n_gen_words):
            # Create input vector
            gen_input_batch = np.zeros((1, n_steps, n_inputs))
            unit_batch = np.zeros((n_steps, n_inputs))
            j = 0
            for idx in input_indices:
                word_vector = np.zeros([vocab_size], dtype=float) # one-hot vector]
                word_vector[idx] = 1
                unit_batch[j] = word_vector
                j+=1
            gen_input_batch[0] = unit_batch

            # Generate output
            y_pred = sesh.run(pred_probs, feed_dict={x: gen_input_batch})
            prediction = tf.squeeze(y_pred) # convert prediction to single vector as in [vocab_size] instead of [1 x vocab_size]
            predicted_idx_scalar_tensor = tf.argmax(prediction)
            predicted_idx = sesh.run(predicted_idx_scalar_tensor) # Evaluate to get tensor value
            predicted_word = ix_to_word[predicted_idx]
            written_words.append(predicted_word)
            # DEBUG
            # print(sesh.run(predicted_idx_scalar_tensor))
            # print(sesh.run(tf.gather(prediction, predicted_idx_scalar_tensor)))
            # print(sesh.run(prediction))

            # Modify input for next iteration
            input_indices = input_indices[1:]
            input_indices.append(predicted_idx)
            #print input_indices

        # Construct text
        txt = ' '.join(written_words)
        print '----\n %s \n----' % (txt, )
elif(isContinuingTraining):
    print("-------- RESUMING TRAINING FROM SERIALIZED POINT --------")
    # Launch the graph
    with tf.Session() as sesh:
        sesh.run(init)
        step = 1

        # Restore model weights from previously saved model
        saver.restore(sesh, model_path)
        print("Model restored from file: %s" % model_path)

        # Train until reach max iterations
        while step * (batch_size/n_steps) < training_iters:
            batch_x, batch_y = next_batch()

            # Run optimization op (backprop)
            sesh.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            # print to the terminal a sample every 100 training steps so we can see how its doing
            if step % 1000 == 0:
                # Calculate batch accuracy
                acc = sesh.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sesh.run(cost, feed_dict={x: batch_x, y: batch_y})
                print "Iter " + str(step*(batch_size/n_steps)) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc)
            step += 1
        print "Optimization Finished!"

        # Save model weights to disk
        save_path = saver.save(sesh, model_path)
        print("Model saved in file: %s" % save_path)
elif(isTextGen):
    print("--------------- TEXT GENERATION ---------------")
    # Launch the graph
    with tf.Session() as sesh:
        sesh.run(init)

        # Restore model weights from previously saved model
        saver.restore(sesh, model_path)
        print("Model restored from file: %s" % model_path)

        # Randomly enerate seed text of 3 words from training data
        rdm_idx1 = word_to_ix[random.choice(data_words)]
        rdm_idx2 = word_to_ix[random.choice(data_words)]
        rdm_idx3 = word_to_ix[random.choice(data_words)]
        input_indices = [rdm_idx1,rdm_idx2,rdm_idx3]
        print("SEED TEXT:")
        txt = ' '.join(ix_to_word[ix] for ix in input_indices)
        print '%s' % (txt, )

        # For each word to be generated
        n_gen_words = args['words']
        written_words = []
        for i in range(n_gen_words):
            # Create input vector
            gen_input_batch = np.zeros((1, n_steps, n_inputs))
            unit_batch = np.zeros((n_steps, n_inputs))
            j = 0
            for idx in input_indices:
                word_vector = np.zeros([vocab_size], dtype=float) # one-hot vector]
                word_vector[idx] = 1
                unit_batch[j] = word_vector
                j+=1
            gen_input_batch[0] = unit_batch

            # Generate output
            y_pred = sesh.run(pred_probs, feed_dict={x: gen_input_batch})
            prediction = tf.squeeze(y_pred) # convert prediction to single vector as in [vocab_size] instead of [1 x vocab_size]
            predicted_idx_scalar_tensor = tf.argmax(prediction)
            predicted_idx = sesh.run(predicted_idx_scalar_tensor) # Evaluate to get tensor value
            predicted_word = ix_to_word[predicted_idx]
            written_words.append(predicted_word)
            # DEBUG
            # print(sesh.run(predicted_idx_scalar_tensor))
            # print(sesh.run(tf.gather(prediction, predicted_idx_scalar_tensor)))
            # print(sesh.run(prediction))

            # Modify input for next iteration
            input_indices = input_indices[1:]
            input_indices.append(predicted_idx)
            #print input_indices

        # Construct text
        txt = ' '.join(written_words)
        print '----\n %s \n----' % (txt, )
