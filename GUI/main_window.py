import PySimpleGUI27 as sg
import os
import io
from os import path

from PIL import Image, ImageTk

sg.ChangeLookAndFeel('BlueMono')

basepath = path.dirname(__file__)
folder = path.abspath(path.join(basepath, "..", "Driving", "generated_inputs"))
img_types = (".png")

def get_imgs(f):
	flist = os.listdir(f)
	fnames = [f for f in flist if os.path.isfile(os.path.join(folder,f)) and f.lower().endswith(img_types)]
	num_files = len(fnames)
	return fnames, num_files

def get_img_data(f, maxsize = (1200,850)):
    img = Image.open(f)
    img.thumbnail(maxsize)
    bio = io.BytesIO()
    img.save(bio, format = "PNG")
    del img
    return bio.getvalue()

fnames, num_files = get_imgs(folder)
filename = os.path.join(folder, fnames[0])
image_elem = sg.Image(data = get_img_data(filename))
filename_display_elem = sg.Text(filename, size=(120,3))
file_num_display_elem = sg.Text('File 1 of {}'.format(num_files), size=(15,1))

col = [[filename_display_elem],
       [image_elem]]

col_files = [[sg.ReadButton('Next', size=(8,2)), sg.ReadButton('Prev', size=(8,2)), file_num_display_elem]]

layout = [[sg.Text('Activation threshold'), sg.Input(size=(5,1),key='actThresh',do_not_clear=True), sg.Text('Behavior differential'), sg.Input(size=(5,1),key='behavDiff',do_not_clear=True)],
          [sg.Text('Number of seeds'), sg.Input(size=(5,1),key='numSeeds',do_not_clear=True), sg.Text('Step size'), sg.Input(size=(5,1),key='stepSize',do_not_clear=True)],
          [sg.Text('Iterations'), sg.Input(size=(5,1),key='numIterations',do_not_clear=True), sg.Text('Gradient ascent'), sg.Input(size=(5,1),key='gradAscent',do_not_clear=True)],
          [sg.Text('Augmentation type'),sg.InputCombo(('light', 'occl', 'blackout', 'blur'), size=(10, 1), readonly=True,key='augType',change_submits=True), sg.Text('Augmentation specific values'),sg.InputCombo(('Scratched lens', 'Raindrops', 'Unfocused camera'), size=(20, 1), readonly=True,key='blurType'),sg.Text('Blur dimension'), sg.Input(size=(5,1),key='blurDimen',do_not_clear=True),sg.Text('Ellipse dimension'), sg.Input(size=(5,1),key='elipDimen',do_not_clear=True)],
          [sg.Text('Testing database'),sg.InputCombo(('Drebin', 'Driving', 'ImageNet', 'MNIST', 'PDF'), size=(10, 1), readonly=True,key='testDb')],
          [sg.Column(col_files),sg.Column(col)],
          [sg.Button('Run simulation')]]

window = sg.Window('DeepXplore GUI').Layout(layout)

#aug_type: light, occl, blackout, blur
#weight_diff: 1
#weight_nc: 0.1 to 0.5
#step: 2 to 20
#seeds: 1-100
#grad_iterations: 20
#treshold: 0

i = 0
while True:
    event, values = window.Read()
    if event is None:
        break
    elif event in ('Next'):
        i += 1
        if i >= num_files:
            i -= num_files
        filename = os.path.join(folder, fnames[i])
	image_elem.Update(data = get_img_data(filename))
	filename_display_elem.Update(filename)
	file_num_display_elem.Update('File {} of {}'.format(i + 1, num_files))
    elif event in ('Prev'):
        i -= 1
        if i < 0:
            i = num_files + i
        filename = os.path.join(folder, fnames[i])
	image_elem.Update(data = get_img_data(filename))
	filename_display_elem.Update(filename)
	file_num_display_elem.Update('File {} of {}'.format(i + 1, num_files))
    elif event in ('Run simulation'):
	os.chdir("..")
	os.chdir(values.get('testDb'))
	os.system('python gen_diff.py %s %d %d %d %d %d %d' % (values.get('augType'), int(window.FindElement('behavDiff').Get()), float(window.FindElement('gradAscent').Get()), int(window.FindElement('stepSize').Get()), int(window.FindElement('numSeeds').Get()), int(window.FindElement('numIterations').Get()), int(window.FindElement('actThresh').Get())))
	i = 0
	folder = path.abspath(path.join(basepath, "..", "%s", "generated_inputs")) % values.get('testDb')
	fnames,num_files = get_imgs(folder)
	filename = os.path.join(folder, fnames[i])
	image_elem.Update(data = get_img_data(filename))
	filename_display_elem.Update(filename)
	file_num_display_elem.Update('File {} of {}'.format(i + 1, num_files))

