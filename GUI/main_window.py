import PySimpleGUI27 as sg

layout = [[sg.Text('Activation treshold'), sg.Input()],
          [sg.Text('Behavior differential'), sg.Input()],
          [sg.Text('Number of seeds'), sg.Input()],
          [sg.Text('Step size'), sg.Input()],
          [sg.Text('Iterations'), sg.Input()],
          [sg.Text('Gradient ascent'), sg.Input()],
          [sg.Submit()]]

event, (aTresh,behavDiff,numSeeds,stepSize,iterations,gradAscent,) = sg.Window('DeepXplore GUI').Layout(layout).Read()

sg.Popup(event, aTresh, behavDiff, numSeeds, stepSize, iterations, gradAscent)
