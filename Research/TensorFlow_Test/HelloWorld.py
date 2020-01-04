import tensorflow as tf
import math


print(('Is your GPU available for use?\n{0}').format(
    'Yes, your GPU is available: True' if tf.test.is_gpu_available() == True else 'No, your GPU is NOT available: False'
))

print(('\nYour devices that are available:\n{0}').format(
    [device.name for device in tf.config.experimental.list_physical_devices()]
))

print(tf.__version__)
hello = tf.constant(name='op1', value='Hello World 1')

a = tf.constant(3)
b = tf.constant(5)
c = a + b
print(c)

mat = tf.constant([[2,3],[1,2],[5,6]])
vec = tf.constant([[1],[2]])

print(mat)

out = tf.matmul(mat, vec)

print(out)

Hola = tf.Variable(5.0)


print(Hola)


# Making a constant tensor A, that does not change
A = tf.constant([[3, 2],
                 [5, 2]])

# Making a Variable tensor VA, which can change. Notice it's .Variable
VA = tf.Variable([[3, 2],
                 [5, 2]])

# Making another tensor B
B = tf.constant([[9, 5],
                 [1, 3]])

# Concatenate columns
AB_concatenated = tf.concat(values=[A, B], axis=1)
print(('Adding B\'s columns to A:\n{0}').format(
    AB_concatenated.numpy()
))

# Concatenate rows
AB_concatenated = tf.concat(values=[A, B], axis=0)
print(('\nAdding B\'s rows to A:\n{0}').format(
    AB_concatenated.numpy()
))

def gelu(x):
    return 0.5*x*(1+tf.tanh(tf.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

def get_gradient(x, activation_funciton):
    with tf.GradientTape() as gt:
        y = activation_funciton(x)
    gradient = gt.gradient(y, x).numpy()
    return gradient

x = tf.Variable(0.5)
gradient = get_gradient(x, gelu)

print('{0} is the gradient of GELU with x={1}'.format(
    gradient, x.numpy()
))

