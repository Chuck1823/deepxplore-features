import PySimpleGUI27 as sg
import os
import io
from PIL import Image, ImageTk

sg.ChangeLookAndFeel('BlueMono')

folder = '/home/charles/deepxplore_senior_design/Driving/generated_inputs'

img_types = (".png")
flist0 = os.listdir(folder)
fnames = [f for f in flist0 if os.path.isfile(os.path.join(folder,f)) and f.lower().endswith(img_types)]
num_files = len(fnames)

def get_img_data(f, maxsize = (1200,850)):
    img = Image.open(f)
    img.thumbnail(maxsize)
    bio = io.BytesIO()
    img.save(bio, format = "PNG")
    del img
    return bio.getvalue()

filename = os.path.join(folder, fnames[0])
image_elem = sg.Image(data = get_img_data(filename))
filename_display_elem = sg.Text(filename, size=(100,3))
file_num_display_elem = sg.Text('File 1 of {}'.format(num_files), size=(15,1))

col = [[filename_display_elem],
       [image_elem]]

col_files = [[sg.ReadButton('Next', size=(8,2)), sg.ReadButton('Prev', size=(8,2)), file_num_display_elem]]

layout = [[sg.Text('Activation treshold'), sg.Input(size=(5,1),key='actTresh',do_not_clear=True), sg.Text('Behavior differential'), sg.Input(size=(5,1),key='behavDiff',do_not_clear=True)],
          [sg.Text('Number of seeds'), sg.Input(size=(5,1),key='numSeeds',do_not_clear=True), sg.Text('Step size'), sg.Input(size=(5,1),key='stepSize',do_not_clear=True)],
          [sg.Text('Iterations'), sg.Input(size=(5,1),key='numIterations',do_not_clear=True), sg.Text('Gradient ascent'), sg.Input(size=(5,1),key='gradAscent',do_not_clear=True)],
          [sg.Text('Augmentation type'),sg.InputCombo(('Lighting', 'Occlusion', 'Blackout', 'Blurring'), size=(10, 1), readonly=True,key='augType',change_submits=True), sg.Text('Augmentation specific values'),sg.InputCombo(('Scratched lens', 'Raindrops', 'Unfocused camera'), size=(20, 1), readonly=True,disabled=True,key='blurType'),sg.Text('Blur dimension'), sg.Input(size=(5,1),key='blurDimen',disabled=True,do_not_clear=True),sg.Text('Ellipse dimension'), sg.Input(size=(5,1),key='elipDimen',disabled=True,do_not_clear=True)],
          [sg.Text('Testing database'),sg.InputCombo(('Drebin', 'Driving', 'ImageNet', 'MNIST', 'PDF'), size=(10, 1), readonly=True,key='testDb')],
          [sg.Column(col_files),sg.Column(col)],
          [sg.Button('Run simulation')]]

window = sg.Window('DeepXplore GUI').Layout(layout)

#aug_type: light occl, blackout
#weight_diff: 1
#weight_nc: 0.1 to 0.5
#step: 2 to 20
#seeds: 1-100
#grad_iterations: 20
#treshold: 0

i = 0
while True:
    event, values = window.Read()
    #print(window.FindElement('numSeeds').Get())
    if event is None:
        break
    elif event in ('Blurring'):
        window.FindElement('blurType').Update(disable=False)
    elif event in ('Lighting'):
        window.FindElement('elipDimen').Update(disable=False)
    elif event in ('Next'):
        i += 1
        if i >= num_files:
            i -= num_files
        filename = os.path.join(folder, fnames[i])
    elif event in ('Prev'):
        i -= 1
        if i < 0:
            i = num_files + i
        filename = os.path.join(folder, fnames[i])
    elif event in ('Run simulation'):
        print("Should call gen_diff.py for db with proper inputs..")
    else:
        filename = os.path.join(folder, fnames[i])

    image_elem.Update(data = get_img_data(filename))
    filename_display_elem.Update(filename)
    file_num_display_elem.Update('File {} of {}'.format(i + 1, num_files))
