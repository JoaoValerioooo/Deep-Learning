import tensorflow as tf
if tf.__version__.startswith('2.'):
    tf = tf.compat.v1
tf.disable_v2_behavior()
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

loss_list_all = []
w_list_all = []
b_list_all = []

#epochs_lim = 350
#epochs_lim = 1000
epochs_lim = 2500
#epochs_lim = 5000
#epochs_lim = 10000
#epochs_lim = 30000

lr = 0.01

for optimizer in (tf.train.MomentumOptimizer(lr, momentum=0.9), tf.train.GradientDescentOptimizer(lr),
                  tf.train.MomentumOptimizer(lr, momentum=0.9, use_nesterov=True), tf.train.AdagradOptimizer(lr),
                  tf.train.AdadeltaOptimizer(lr), tf.train.AdamOptimizer(lr), tf.train.RMSPropOptimizer(lr),
                  tf.train.FtrlOptimizer(learning_rate=lr, l1_regularization_strength=0.001, l2_regularization_strength=0.001),
                  tf.train.ProximalAdagradOptimizer(learning_rate=lr, l1_regularization_strength=0.0001)):

  # Model parameters
  W = tf.Variable([.3], dtype=tf.float32)
  b = tf.Variable([-.3], dtype=tf.float32)
  # Model input and output
  x = tf.placeholder(tf.float32)
  linear_model = W * x + b
  y = tf.placeholder(tf.float32)

  # loss
  loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
  # optimizer
  #optimizer = tf.train.GradientDescentOptimizer(lr)
  train = optimizer.minimize(loss)

  # training data
  x_train = [1, 2, 3, 4]
  y_train = [0, -1, -2, -3]
  # training loop
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init) # reset values to wrong

  curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
  #print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

  loss_list = []
  epoch = []
  w_list = []
  b_list = []
  for i in range(epochs_lim):
    sess.run(train, {x: x_train, y: y_train})
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    #print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    loss_list.append(curr_loss)
    w_list.append(curr_W)
    b_list.append(curr_b)
    epoch.append(i)

  loss_list_all.append(loss_list)
  w_list_all.append(w_list)
  b_list_all.append(b_list)

name = ['Momentum - 0.9', 'GradientDescentOptimizer', 'NesterovAccelerated(NAG) - 0.9', 'Adagrad', 'Adadelta','Adam', 'RMSprop', 'Ftrl', 'ProximalAdagrad']

# Plot Loss
plt.figure()
for i in range(len(name)):
  if i == 1:
    plt.plot(epoch, loss_list_all[1], color="C"+str(i), linestyle="-", label='GradientDescentOptimizer')
    continue
  plt.plot(epoch, loss_list_all[i], color="C"+str(i), linestyle="-", label=name[i])
plt.title("Optimizers LR=0.01 - Epochs="+str(epochs_lim))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='center right')
plt.savefig('Loss_'+str(epochs_lim)+'.png')

# Plot Loss (Log scale)
plt.figure()
for i in range(len(name)):
  if i == 1:
    plt.plot(epoch, loss_list_all[1], color="C"+str(i), linestyle="-", label='GradientDescentOptimizer')
    continue
  plt.semilogy(epoch, loss_list_all[i], color="C"+str(i), linestyle="-", label=name[i])
plt.title("Logaritm - Optimizers LR=0.01 - Epochs="+str(epochs_lim))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='center right')
plt.savefig('Log_Loss_'+str(epochs_lim)+'.png')


# Plot W
plt.figure()
for i in range(len(name)):
  plt.plot(epoch, w_list_all[i], color="C"+str(i), linestyle="-", label=name[i])
plt.title("Optimizers LR=0.01 - Epochs="+str(epochs_lim))
plt.xlabel('Epochs')
plt.ylabel('W')
plt.legend(loc='center right')
plt.savefig('W_'+str(epochs_lim)+'.png')

# Plot b
plt.figure()
for i in range(len(name)):
  plt.plot(epoch, b_list_all[i], color="C"+str(i), linestyle="-", label=name[i])
plt.title("Optimizers LR=0.01 - Epochs="+str(epochs_lim))
plt.xlabel('Epochs')
plt.ylabel('b')
plt.legend(loc='center right')
plt.savefig('b_'+str(epochs_lim)+'.png')