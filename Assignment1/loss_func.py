import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    V = inputs
    U = true_w
    dot_prod = tf.multiply(U,V)
    #print('dot prod shape ',tf.shape(dot_prod))
    A = tf.reduce_sum(dot_prod,1)
    #print('A shape ', tf.shape(A), 'batch size = ')
    VU = tf.matmul(V,U,transpose_b=True)
    B = tf.log(tf.reduce_sum(tf.exp(VU),1))
    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weights: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimension is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    V = inputs
    Uo = tf.reshape(tf.nn.embedding_lookup(weights, labels), (-1,128))
    bo = tf.nn.embedding_lookup(biases, labels)
    po = tf.gather(unigram_prob, labels)
    Un = tf.reshape(tf.nn.embedding_lookup(weights, sample), (-1,128))
    bn = tf.nn.embedding_lookup(biases, sample)
    pn = tf.gather(unigram_prob, sample)
    #with tf.Session() as ses:
    #    print(ses.run([tf.shape(V), tf.shape(Uo), tf.shape(Un), tf.shape(bo), tf.shape(bn), tf.shape(po), tf.shape(pn)]))
    dot_prod = tf.multiply(V,Uo)
    #print(sample.__class__)
    k = sample.shape[0]
    A = tf.log(1e-7 + tf.sigmoid(tf.reduce_sum(dot_prod,1) + bo - tf.log(k*po)))
    B1 = tf.log(1e-7 + 1 - tf.sigmoid(tf.matmul(V, tf.transpose(Un)) + bn - tf.log(k*pn)))
    B = tf.reduce_sum(B1,1)

    return tf.add(-A,-B)