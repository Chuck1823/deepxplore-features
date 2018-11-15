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

layout = [[sg.Text('Activation treshold'), sg.Input(size=(5,1)), sg.Text('Behavior differential'), sg.Input(size=(5,1))],
          [sg.Text('Number of seeds'), sg.Input(size=(5,1)), sg.Text('Step size'), sg.Input(size=(5,1))],
          [sg.Text('Iterations'), sg.Input(size=(5,1)), sg.Text('Gradient ascent'), sg.Input(size=(5,1))],
          [sg.Text('Testing database'),sg.InputCombo(('Drebin', 'Driving', 'ImageNet', 'MNIST', 'PDF'), size=(10, 1), readonly=True)],
          [sg.Column(col_files),sg.Column(col)],
          [sg.Button('Run simulation')]]

window = sg.Window('DeepXplore GUI').Layout(layout)

#event, (actTresh, behavDiff, numSeeds, stepSize, numIt, gradAscent, testDb) = window.Read()

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
    elif event in ('Prev'):
        i -= 1
        if i < 0:
            i = num_files + i
        filename = os.path.join(folder, fnames[i])
    else:
        filename = os.path.join(folder, fnames[i])

    image_elem.Update(data = get_img_data(filename))
    filename_display_elem.Update(filename)
    file_num_display_elem.Update('File {} of {}'.format(i + 1, num_files))
