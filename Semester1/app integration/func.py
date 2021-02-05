import os
import builtins
import tkinter as tk
from tkinter import filedialog

from main import *

extensions=[("Vdo files", ".mp4 .MOV")]

def open_file():
    path = filedialog.askopenfilenames(parent=builtins.window,title='Choose a file',filetypes = extensions)
    builtins.path = path
    for e in path:
        builtins.text_widget.insert(tk.END,e+'\n')

def open_folder():
    path = filedialog.askdirectory(parent=builtins.window,title='Choose a file')
    files = [f for f in os.listdir(path)]
    builtins.path = files
    for e in files:
        builtins.text_widget.insert(tk.END,e+'\n')

def proceed(willshowvdo):
    try:
        box_result,imgs,counter = start_inference(builtins.path,willshowvdo)
        for e in counter:
            builtins.text_widget.insert(tk.END,'Frame :'+str(e)+'\n')
        #openNewWindow()
    except Exception as ec:
        print(ec)

def openNewWindow(): 
      
    # Toplevel object which will  
    # be treated as a new window 
    newWindow = tk.Toplevel(builtins.window) 
  
    # sets the title of the 
    # Toplevel widget 
    newWindow.title("New Window") 
  
    # sets the geometry of toplevel 
    newWindow.geometry("200x200") 
  
    # A Label widget to show in toplevel 
    tk.Label(newWindow,text ="This is a new window").pack() 

