import tensorflow as tf

y_hat = tf.constant(36, name='y_hat')
y = tf.constant(39, name='y')

loss = tf.Variable((y-y_hat) ** 2 , name='loss')
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run()
    print(session.run(loss))