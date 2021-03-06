import numpy as np
import nltk
import pickle
import random
import argparse

# parse cli arguments
ap = argparse.ArgumentParser(description="Minimal RNN for text generation")
ap.add_argument('-t', '--train', help = 'Set this flag to train the RNN', action='store_true', default=False)
ap.add_argument('-t2', '--train2', help = 'Set this flag to train the RNN from the last point', action='store_true', default=False)
ap.add_argument('-f', '--file', help = 'The seed file for training or text generation', required=False)
args = vars(ap.parse_args())
filename = args['file']
isTrainingPhase = args['train']
isContinuingTraining = args['train2']
if(args['file']):
    filename = args['file']
else:
    filename = 'fanfic2.txt'


if(isTrainingPhase):
    print("-------- TRAINING --------")
    # data I/O
    data_file = filename
    data = open(data_file, 'r').read()
    #print data
    data_words = nltk.word_tokenize(data.decode('utf8'))
    # Make input into a set to remove duplicates and then make it into a list
    words = list(set(data_words))
    data_size, vocab_size = len(data_words), len(words)
    print 'Data has %d characters, %d unique.' % (data_size,vocab_size)

    # Create dictionaries mapping a word to an index and vice versa
    word_to_ix = { ch:i for i,ch in enumerate(words) }
    ix_to_word = { i:ch for i,ch in enumerate(words) }

    # hyperparameters
    hidden_size = 100 # size of hidden layer of neurons
    seq_length = 25 # number of steps to unroll the RNN for
    learning_rate = 1e-1

    # initialise model parameters
    Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
    Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
    Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
    bh = np.zeros((hidden_size, 1)) # hidden bias
    by = np.zeros((vocab_size, 1)) # output bias

    # Initialise rnn for serialization
    rnn = {}
    rnn["hidden_size"] = hidden_size
    rnn["seq_length"] = seq_length
    rnn["learning_rate"] = learning_rate


    def lossFun(inputs, targets, hprev):
        """
        inputs,targets are both list of integers.
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        # dictionaries for values at each timestep indexed by timestep
        # xs[t]-> input_t, hs[t]->hiddenState_t, ys[t]->output_t, ps[t]->probabilities_t,
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0

        # FORWARD PASS
        # Go through each timestep t
        for t in xrange(len(inputs)):
            # encode input in one-hot encoding (aka 1-of-k encoding)
            xs[t] = np.zeros((vocab_size,1))
            xs[t][inputs[t]] = 1
            # Update our hidden state according to the recurrent function f_W(x_t, h_t-1)
            # given as h_t = tanh(W_xh.x_t + W_hh.h_t-1 + bias)
            hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state

            # Compute our output
            ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
            # Accumulate the loss for this time step as the negative log of the predicted probability.
            # Ideally, we would have a probability of 1 for the actual next character. If it is 1, the loss is 0, log(1) = 0.
            loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)


        # BACKWARD PASS: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why) # Initialise gradients of weight matrices
        dbh, dby = np.zeros_like(bh), np.zeros_like(by) # Initialise gradients of biases
        dhnext = np.zeros_like(hs[0]) # Initialise gradient for next timestep
        for t in reversed(xrange(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(Why.T, dy) + dhnext # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

        rnn["loss"] = loss
        rnn["dWxh"] = dWxh
        rnn["dWhh"] = dWhh
        rnn["dWhy"] = dWhy
        rnn["dbh"] = dbh
        rnn["dby"] = dby
        rnn["Wxh"] = Wxh
        rnn["Whh"] = Whh
        rnn["Why"] = Why
        rnn["bh"] = bh
        rnn["by"] = by
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

    def sample(h, seed_ix, n):
        """
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter for first time step
        """
        # Set up our one-hot encoded input vector based on the seed character.
        x = np.zeros((vocab_size, 1))
        x[seed_ix] = 1

        # Set up an array to keep track of our sequence.
        ixes = []

        # For each timestep
        for t in xrange(n):
            # Update hidden state and generate output and apply softmax to get probabilities
            h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
            y = np.dot(Why, h) + by
            p = np.exp(y) / np.sum(np.exp(y))

            # Select the index of the character with the highest probability
            ix = np.random.choice(range(vocab_size), p=p.ravel())

            # Create new one-hot encoding input for selected character
            x = np.zeros((vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes


    # TRAINING
    # n is the number of training iterations we've done. p is the index into our training data for where we are now.
    n, p = 0, 0

    # Set up memory variables for the Adagrad algorithm
    mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
    smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

    # Trraining loop
    try:
        while True:
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p+seq_length+1 >= len(data_words) or n == 0:
                hprev = np.zeros((hidden_size,1)) # reset RNN memory
                p = 0 # go from start of data

            # Fetch inputs and targets of length seq_length at a time
            inputs = [word_to_ix[w] for w in data_words[p:p+seq_length]]
            # we're predicting the next character so the target for data[i] is data[i+1]
            targets = [word_to_ix[w] for w in data_words[p+1:p+seq_length+1]]

            # print to the terminal a sample every 100 training steps so we can see how its doing
            if n % 100 == 0:
                sample_ix = sample(hprev, inputs[0], 200)
                txt = ' '.join(ix_to_word[ix] for ix in sample_ix)
                print '----\n %s \n----' % (txt, )

            # forward seq_length characters through the net and fetch gradient
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001 # Adagrad stuff
            if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress

            # perform parameter update with Adagrad
            for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                        [dWxh, dWhh, dWhy, dbh, dby],
                                        [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

            p += seq_length # move data pointer tonext chunk of size seq_length
            n += 1 # iteration counter
            rnn["h"] = hprev
            rnn["nuber_of_iterations"] = n
            rnn["position_in_data"] = p
    except KeyboardInterrupt:
        #Serialize hidden state to use for prediction later
        f = open('rnn.ser', 'wb')
        pickle.dump(rnn, f)
        f.close()
        exit()
elif(isContinuingTraining):
    print("-------- RESUMING TRAINING FROM SERIALIZED POINT --------")
    # Load serialized rnn parameters
    f = open("rnn.ser", 'rb')
    rnn = pickle.load(f)
    f.close()

    # data I/O
    data_file = filename
    data = open(data_file, 'r').read()
    #print data
    data_words = nltk.word_tokenize(data.decode('utf8'))
    # Make input into a set to remove duplicates and then make it into a list
    words = list(set(data_words))
    data_size, vocab_size = len(data_words), len(words)
    print 'Data has %d characters, %d unique.' % (data_size,vocab_size)

    # Create dictionaries mapping a word to an index and vice versa
    word_to_ix = { ch:i for i,ch in enumerate(words) }
    ix_to_word = { i:ch for i,ch in enumerate(words) }

    # hyperparameters
    hidden_size = 100 # size of hidden layer of neurons
    seq_length = 25 # number of steps to unroll the RNN for
    learning_rate = 1e-1

    # Model weights and biases
    Wxh = rnn["Wxh"]
    Whh = rnn["Whh"]
    Why = rnn["Why"]
    bh = rnn["bh"]
    by = rnn["by"]
    hprev = rnn["h"]

    # Load rnn parameters
    hidden_size = rnn["hidden_size"]
    seq_length = rnn["seq_length"]
    learning_rate = rnn["learning_rate"]


    def lossFun(inputs, targets, hprev):
        """
        inputs,targets are both list of integers.
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        # dictionaries for values at each timestep indexed by timestep
        # xs[t]-> input_t, hs[t]->hiddenState_t, ys[t]->output_t, ps[t]->probabilities_t,
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0

        # FORWARD PASS
        # Go through each timestep t
        for t in xrange(len(inputs)):
            # encode input in one-hot encoding (aka 1-of-k encoding)
            xs[t] = np.zeros((vocab_size,1))
            xs[t][inputs[t]] = 1
            # Update our hidden state according to the recurrent function f_W(x_t, h_t-1)
            # given as h_t = tanh(W_xh.x_t + W_hh.h_t-1 + bias)
            hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state

            # Compute our output
            ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
            # Accumulate the loss for this time step as the negative log of the predicted probability.
            # Ideally, we would have a probability of 1 for the actual next character. If it is 1, the loss is 0, log(1) = 0.
            loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)


        # BACKWARD PASS: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why) # Initialise gradients of weight matrices
        dbh, dby = np.zeros_like(bh), np.zeros_like(by) # Initialise gradients of biases
        dhnext = np.zeros_like(hs[0]) # Initialise gradient for next timestep
        for t in reversed(xrange(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(Why.T, dy) + dhnext # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

        rnn["loss"] = loss
        rnn["dWxh"] = dWxh
        rnn["dWhh"] = dWhh
        rnn["dWhy"] = dWhy
        rnn["dbh"] = dbh
        rnn["dby"] = dby
        rnn["Wxh"] = Wxh
        rnn["Whh"] = Whh
        rnn["Why"] = Why
        rnn["bh"] = bh
        rnn["by"] = by
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

    def sample(h, seed_ix, n):
        """
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter for first time step
        """
        # Set up our one-hot encoded input vector based on the seed character.
        x = np.zeros((vocab_size, 1))
        x[seed_ix] = 1

        # Set up an array to keep track of our sequence.
        ixes = []

        # For each timestep
        for t in xrange(n):
            # Update hidden state and generate output and apply softmax to get probabilities
            h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
            y = np.dot(Why, h) + by
            p = np.exp(y) / np.sum(np.exp(y))

            # Select the index of the character with the highest probability
            ix = np.random.choice(range(vocab_size), p=p.ravel())

            # Create new one-hot encoding input for selected character
            x = np.zeros((vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes


    # TRAINING
    # n is the number of training iterations we've done. p is the index into our training data for where we are now.
    n = rnn["nuber_of_iterations"]
    p = rnn["position_in_data"]

    # Set up memory variables for the Adagrad algorithm
    mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
    smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

    # Training loop
    print "ITERTIONS: %d , DATA POS: %d" % (n,p)
    try:
        while True:
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p+seq_length+1 >= len(data_words) or n == 0:
                hprev = np.zeros((hidden_size,1)) # reset RNN memory
                p = 0 # go from start of data

            # Fetch inputs and targets of length seq_length at a time
            inputs = [word_to_ix[w] for w in data_words[p:p+seq_length]]
            # we're predicting the next character so the target for data[i] is data[i+1]
            targets = [word_to_ix[w] for w in data_words[p+1:p+seq_length+1]]

            # print to the terminal a sample every 100 training steps so we can see how its doing
            if n % 100 == 0:
                sample_ix = sample(hprev, inputs[0], 200)
                txt = ' '.join(ix_to_word[ix] for ix in sample_ix)
                print '----\n %s \n----' % (txt, )

            # forward seq_length characters through the net and fetch gradient
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001 # Adagrad stuff
            if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress

            # perform parameter update with Adagrad
            for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                        [dWxh, dWhh, dWhy, dbh, dby],
                                        [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

            p += seq_length # move data pointer tonext chunk of size seq_length
            n += 1 # iteration counter
            rnn["h"] = hprev
            rnn["nuber_of_iterations"] = n
            rnn["position_in_data"] = p
    except KeyboardInterrupt:
        #Serialize hidden state to use for prediction later
        f = open('rnn.ser', 'wb')
        pickle.dump(rnn, f)
        f.close()
        exit()
else:
    # data I/O
    data_file = filename
    data = open(data_file, 'r').read()
    #print data
    data_words = nltk.word_tokenize(data.decode('utf8'))
    # Make input into a set to remove duplicates and then make it into a list
    words = list(set(data_words))
    data_size, vocab_size = len(data_words), len(words)
    print 'Data has %d characters, %d unique.' % (data_size,vocab_size)

    # Load serialized rnn parameters
    f = open("rnn_jaden.ser", 'rb')
    rnn = pickle.load(f)
    f.close()

    # model hyperparameters
    hidden_size = rnn["hidden_size"]
    seq_length = rnn["seq_length"]
    seq_length = rnn["learning_rate"]

    # Create dictionaries mapping a word to an index and vice versa
    word_to_ix = { ch:i for i,ch in enumerate(words) }
    ix_to_word = { i:ch for i,ch in enumerate(words) }

    # Model weights and biases
    Wxh = rnn["Wxh"]
    Whh = rnn["Whh"]
    Why = rnn["Why"]
    bh = rnn["bh"]
    by = rnn["by"]

    h = rnn["h"]

    def sample(h, seed_ix, n):
        """
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter for first time step
        """
        # Set up our one-hot encoded input vector based on the seed character.
        x = np.zeros((vocab_size, 1))
        x[seed_ix] = 1

        # Set up an array to keep track of our sequence.
        ixes = []

        # For each timestep
        for t in xrange(n):
            # Update hidden state and generate output and apply softmax to get probabilities
            h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
            y = np.dot(Why, h) + by
            p = np.exp(y) / np.sum(np.exp(y))

            # Select the index of the character with the highest probability
            ix = np.random.choice(range(vocab_size), p=p.ravel())

            # Create new one-hot encoding input for selected character
            x = np.zeros((vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

    # Sample and generate words
    sample_ix = sample(h, word_to_ix[random.choice(data_words)], 200)
    txt = ' '.join(ix_to_word[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )
