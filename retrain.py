import tensorflow as tf,sys
retrain_data_path = sys.argv[1]

#read in the data
retrain_data = tf.gfile.FastGFile(retrain_data_path, 'rb').read()

#loads label data, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("/tf_files/retrained_labels.txt")]

# unpersists graph from file
with tf.gfile.FastGfile("/tf_files/retrained_graph.pb", 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  _ = tf.import_graph_def(graph_def, name='')
with tf.Session() as sess:
  #feed the image data as input to the graph and get first prediction
  softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
  predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0' : retrain_data})
  
  # sort to show labels of first prediction in order of confidence
  top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
  for node_id in top_k:
    human_string = label_lines[node_id]
    score = predictions[0][node_id]
    print('%s (score=%s.5f)' & (human_string, score))
