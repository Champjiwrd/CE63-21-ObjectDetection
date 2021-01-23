#Any dependencies
import tkinter as tk
from tkinter import filedialog

#init
window = tk.Tk()
window.title("Example for Tkinter")  # to define the title

#size
canvas = tk.Canvas(window, width=450, height=500)  # define the size
canvas.pack()

def open_file():
    file = filedialog.askopenfile(parent=window,mode='rb',title='Choose a file')
    if file is not None: 
        content = file.read() 
        print(content)

btn = tk.Button(window, text ='Browse', command = lambda:open_file() ,activebackground = 'gray') 
btn.pack(pady = 10) 


#Running
window.mainloop()