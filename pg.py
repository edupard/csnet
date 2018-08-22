import numpy as np

import tensorflow as tf

w = 100
h = 100
c = 32

tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))


palette = tf.constant(np.zeros((c, 3)))

policy_logits = tf.contrib.eager.Variable(np.ones((w,h,c), dtype=np.float32))

flat_policy_logits = tf.reshape(policy_logits, (-1, c))

flat_action_samples = tf.multinomial(flat_policy_logits, 1)
action_sample = tf.reshape(flat_action_samples, (w,h, 1))

picture = tf.gather_nd(palette, action_sample)


updates = tf.ones((w * h))
indices = action_sample
scatter = tf.scatter_nd(indices, updates, (w,h,c))





#tf.nn.softmax_cross_entropy_with_logits_v2(logits=policy_logits, labels=)

i = 0


