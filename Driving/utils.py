import cv2
import math
import random
from collections import defaultdict
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.preprocessing import image
from scipy.misc import imsave



def draw_arrow(img, angle1, angle2, angle3):
    pt1 = (img.shape[1] / 2, img.shape[0])
    pt2_angle1 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle1)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle1)))
    pt2_angle2 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle2)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle2)))
    pt2_angle3 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle3)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle3)))
    img = cv2.arrowedLine(img, pt1, pt2_angle1, (0, 0, 255), 1)
    img = cv2.arrowedLine(img, pt1, pt2_angle2, (0, 255, 0), 1)
    img = cv2.arrowedLine(img, pt1, pt2_angle3, (255, 0, 0), 1)
    return img


def angle_diverged(angle1, angle2, angle3):
    if (abs(angle1 - angle2) > 0.2 or abs(angle1 - angle3) > 0.2 or abs(angle2 - angle3) > 0.2) and not (
                (angle1 > 0 and angle2 > 0 and angle3 > 0) or (
                                angle1 < 0 and angle2 < 0 and angle3 < 0)):
        return True
    return False


def preprocess_image(img_path, target_size=(100, 100)):
    img = image.load_img(img_path, target_size=target_size)
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data


def deprocess_image(x):
    x = x.reshape((100, 100, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def atan_layer(x):
    return tf.multiply(tf.atan(x), 2)


def atan_layer_shape(input_shape):
    return input_shape


def normal_init(shape):
    return K.truncated_normal(shape, stddev=0.1)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads

def constraint_rain(gradients, img, size=30):
    start_point = (
        random.randint(0, gradients.shape[1] - size), random.randint(0, gradients.shape[2] - size))
    image = Image.open(img)
    cropped_image = image.crop((start_point[0],start_point[1],start_point[0]+size,start_point[1]+size))
    blurred_image = cropped_image.filter(ImageFilter.GaussianBlur(radius=5))
    image.paste(blurred_image,(start_point[0],start_point[1],start_point[0]+size,start_point[1]+size))
    image.save('./generated_inputs/temp.png')

def constraint_bright(gradients, img, size=30):
    start_point = (
        random.randint(0, gradients.shape[1] - size), random.randint(0, gradients.shape[2] - size))
    image = Image.open(img)
    white = Image.new('RGB',(gradients.shape[1],gradients.shape[2]),'#ffffff')
    cropped_image = white.crop((start_point[0],start_point[1],start_point[0]+size,start_point[1]+size))
    image.paste(cropped_image,(start_point[0],start_point[1],start_point[0]+size,start_point[1]+size))
    image.save('./generated_inputs/temp.png')

def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = 500 * np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(10, 10)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    # if np.mean(patch) < 0:
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


def init_coverage_tables(model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in xrange(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True


def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False


def update_heatmap(orig_img, aug_img, heatmap):
    for x in xrange(heatmap.shape[0]):
        for y in xrange(heatmap.shape[1]):
            if orig_img[0,x,y,0] != aug_img[0,x,y,0]:
                heatmap[x,y] += 1
    hm_avg = np.mean(heatmap)
    hm_colored = np.zeros([heatmap.shape[0], heatmap.shape[1], 3], dtype=np.uint8)
    for x in xrange(heatmap.shape[0]):
        for y in xrange(heatmap.shape[1]):
            if heatmap[x,y] >= hm_avg + .25 * hm_avg:
                hm_colored[x,y] = [255,0,0]
            elif heatmap[x,y] >= hm_avg and heatmap[x,y] < hm_avg + .25 * hm_avg:
                hm_colored[x,y] = [255,165,0]
            elif heatmap[x,y] < hm_avg and heatmap[x,y] >= hm_avg - .25 * hm_avg:
                hm_colored[x,y] = [0,255,0]
            else:
                hm_colored[x,y] = [0,0,255]
    print('updated heatmap')
    return heatmap, hm_colored

def save_heatmap(hm, aug, num_imgs):
    fn = "MNIST_" + aug + "_" + str(num_imgs)
    fp = "./heatmaps/" + fn + ".pdf"
    pp = PdfPages(fp)
    fig = plt.figure(figsize = (8.5, 11))
    plt.imshow(hm)
    pp.savefig()
    plt.close()
    pp.close()
    print('heatmap saved to ' + fp)

def error_pattern_match(hm, orig_img_list, gen_img_list, transformation, p1, p2 ,p3):
    error_pattern_set = []
    p1_error = []
    p2_error = []
    p3_error = []
    for i, img in enumerate(gen_img_list):
        done = False
        for x in xrange(hm.shape[0]):
            for y in xrange(hm.shape[1]):
                pixel = hm[x,y]
                orig_img = orig_img_list[i]
                if orig_img[0,x,y,0] != img[0,x,y,0] and pixel[0] == 255:
                    error_pattern_set.append(draw_arrow(deprocess_image(img),p1[i],p2[i],p3[i]))
                    p1_error.append(str(p1[i]))
                    p2_error.append(str(p1[i]))
                    p3_error.append(str(p3[i]))
                    done = True
                    break
            if done:
                break

    for i, img in enumerate(error_pattern_set):
        imsave('./error_pattern_set/' + transformation + '_' + p1_error[i] + '_' + p2_error[i] +
                '_' + p3_error[i] + '.png', img)
    print("Error pattern set saved to ./error_pattern_set/ folder")

def make_scatter_plot(scatter_plot_data, aug, num_imgs):

    data = []

    for i in range(len(scatter_plot_data)):
        for j in range(3):
            data.append((i,scatter_plot_data[i][j-1]))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for pair in data:
        x,y = pair
        ax.scatter(x,y,alpha=0.8,c='red')
    title = 'Driving_' + str(aug) + '_' + str(num_imgs)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Predictions')
    #plt.show()

    print ('update scatter plot')
    return fig

def save_scatter_plot(scatter_plot, aug, num_imgs):
    fn = "MNIST_" + aug + "_" + str(num_imgs)
    fp = "./scatterplots/" +fn + ".pdf"
    scatter_plot.savefig(fn,bbox_inches='tight')
    pp=PdfPages(fn)
    pp.savefig()
    pp.close()

    print('Scatter plot saved to: '+ fp)
