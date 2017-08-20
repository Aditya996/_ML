import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tx=[2,4,6,8,10,12,14,16,18]
ty=[8,8,9,4,15,11,14,17,12]
W = tf.Variable([3], dtype=tf.float64)
b = tf.Variable([9], dtype=tf.float64)

x = tf.placeholder(tf.float64)
linear_model = W * x + b
y = tf.placeholder(tf.float64)

loss = (tf.reduce_sum(tf.square(linear_model - y)))/(2*len(tx))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
cnt=0
for i in range(10000):
  sess.run(train, {x:tx, y:ty})
  if(i%1000==0):
      cnt+=1
      print(cnt)
      print('W:',sess.run([W]))
      
res_W, res_b, res_loss = sess.run([W, b, loss], {x:tx, y:ty})
print("W: %s b: %s loss: %s"%(res_W, res_b, res_loss))
ip=int(input("enter x to predict y : "))
op=(float(res_W)*ip)+float(res_b)
print('x:',ip,',y:',op)
op_liney=sess.run(linear_model,{x:tx})
plt.plot(tx,op_liney)
plt.plot(ip,op,'r*')
plt.plot(tx,ty,'g^')
plt.show()
