#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re
import tensorflow as tf

#read data from csv file
df = pd.read_csv('train.csv')
#change data to array
data = np.asarray(df.values)

class tagquora:
    def __init__(self, train_file_path=None, test_file_path=None, dim_embeded=64, rnn_size=32, layer_size=2, batch_size=128):
        self.train_file_path=train_file_path
        self.test_file_path=test_file_path
        self.dim_embeded=dim_embeded
        self.rnn_size=rnn_size
        self.layer_size=layer_size
        self.batch_size=batch_size

    def load_file(self, file=None):
        print("load file from " + file)
        if len(file.split('csv')) > 0:
            # read csv file
            csv_data = pd.read_csv(file)
            data = np.asarray(csv_data.values)
        else:
            # read file
            data = []
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line_data = line.split()
                    data.append(line_data)
            data = np.asarray(data)

        return data

    def generate_dict(self, data):
        vocabulary_dict = {}
        vocabulary_dict["UNK"] = 0;
        max_len = 0

        for item in data:
            question = item[1]
            clear_text = re.sub(r'[\?]|[,]|[/]|["]|[(]|[)]|[.]|[!]', " ", question)
            clear_text = clear_text.strip()
            items =re.split(r'\s+', clear_text)
            if len(items) > max_len:
                max_len = len(items)
            for word in set(items):
                if word not in vocabulary_dict:
                    vocabulary_dict[word] = len(vocabulary_dict)

        print("vocabulary size is " + str(len(vocabulary_dict)))

        return vocabulary_dict, max_len

    def generate_next_batch(self):
        while True:
            id = []
            x_batch = []
            y_batch = []
            len_batch = []
            for i in range(data.shape[0]):
                item = data[i]
                id.append(item[0])
                question = item[1]
                clear_text = re.sub(r'[\?]|[,]|[/]|["]|[(]|[)]|[.]|[!]', " ", question)
                clear_text = clear_text.strip()
                items =re.split(r'\s+', clear_text)
                x = [self.vocabulary_dict[i] for i in items]
                if len(x) < self.max_len:
                    length = len(x)
                    x[length:] = [0 for i in range(0, self.max_len - length)]
                x_batch.append(x)
                if len(item) > 2:
                    y_batch.append(item[2])
                len_batch.append(length)

                if len(x_batch)%self.batch_size == 0:
                    yield id, np.asarray(x_batch), np.asarray(y_batch), np.asarray(len_batch)
                    id = []
                    x_batch = []
                    y_batch = []
                    len_batch = []

    def model(self):
        x_input = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_len], name="x_input")
        y_input = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, 1], name="y_input")

def generate_dict(data):
    vocabulary_dict = {}
    vocabulary_dict["UNK"] = 0;
    max_len = 0

    for item in data:
        question = item[1]
        clear_text = re.sub(r'[\?]|[,]|[/]|["]|[(]|[)]|[.]|[!]', " ", question)
        clear_text = clear_text.strip()
        items =re.split(r'\s+', clear_text)
        if len(items) > max_len:
            max_len = len(items)
        for word in set(items):
            if word not in vocabulary_dict:
                vocabulary_dict[word] = len(vocabulary_dict)

    return vocabulary_dict, max_len


vocabulary_dict, max_len = generate_dict(data)

def get_batch(data):
    id = []
    x_batch = []
    y_batch = []
    len_batch = []
    for item in data:
        id.append(item[0])
        question = item[1]
        clear_text = re.sub(r'[\?]|[,]|[/]|["]|[(]|[)]|[.]|[!]', " ", question)
        clear_text = clear_text.strip()
        items =re.split(r'\s+', clear_text)
        x = [vocabulary_dict[i] for i in items]
        if len(x) < max_len:
            length = len(x)
            x[length:] = [0 for i in range(0, max_len - length)]
        x_batch.append(x)
        y_batch.append(item[2])
        len_batch.append(length)

    return id, np.asarray(x_batch), np.asarray(y_batch), np.asarray(len_batch)

# id, x, y, length = get_batch(data[0:50])

def model():
    batch_size = 64
    vocabulary_size = len(vocabulary_dict)
    embed_dem = 128
    cell_size = 64
    layer_size = 2
    max_step = int(data.shape[0]/batch_size)

    graph = tf.Graph()

    with graph.as_default():
        x_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, max_len])
        y_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, ])
        len_input = tf.placeholder(dtype=tf.int32, shape=[batch_size])
        id_input = tf.placeholder(dtype=tf.string, shape=[batch_size, ])
        with tf.device('/gpu:0'):
            embed = tf.Variable(tf.random_uniform(shape=[vocabulary_size, embed_dem], minval=-1, maxval=1,dtype=tf.float32, name="embed"))
            input_embedding = tf.nn.embedding_lookup(embed, x_input, name="input_embeddings")

            weight_1 = tf.Variable(tf.random_uniform(shape=[2 * cell_size, cell_size], minval=-1, maxval=1,dtype=tf.float32, name="weight_1"))

            weight_2 = tf.Variable(tf.random_uniform(shape=[cell_size, 2], minval=-1, maxval=1,dtype=tf.float32, name="weight_2"))

            bias = tf.get_variable(shape=[batch_size, 1], dtype=tf.float32, name="bias")

        cell = tf.nn.rnn_cell.BasicLSTMCell

        cell_fw = [cell(cell_size) for _ in range(layer_size)]
        cell_bw = [cell(cell_size) for _ in range(layer_size)]

        output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cell_fw,
            cells_bw=cell_bw,
            inputs=input_embedding,
            dtype=tf.float32,
            sequence_length=len_input)

        output = tf.stack(output)

        size = tf.shape(output)[0]
        index = tf.range(0, size) * max_len + (len_input - 1)

        output = tf.gather(tf.reshape(output, [-1, cell_size * 2]), index)

        relias = tf.matmul(output, weight_1)

        logits = tf.matmul(relias, weight_2) + bias

        y_ = tf.nn.softmax(logits)

        pre_y = tf.argmax(y_, 1, output_type=tf.int32)

        accuracy = tf.equal(pre_y, y_input)
        accuracy_op = tf.reduce_mean(tf.cast(accuracy, tf.float32))

        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input))
#         optimizer_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss_op)
        optimizer_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_op)

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as session:
        init.run()
        for epoch in range(0, 10):
            print("epoch is " + str(epoch))
            for step in range(0, max_step):
                start = step * batch_size
                end = (step + 1) * batch_size
                id, x, y, length = get_batch(data[start:end])
                soft, ac, loss, _ = session.run([pre_y, accuracy_op, loss_op, optimizer_op], feed_dict={x_input: x, y_input: y, len_input: length})
                if step % 100 == 0:
                    print("loss is " + str(loss))

model()


# In[ ]:




