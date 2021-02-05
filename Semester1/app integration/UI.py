#Any dependencies
import os
import tkinter as tk
from tkinter import filedialog

import builtins
from func import *


#init
window = tk.Tk() #root
window.title("Example")  # to define the title
window.geometry('800x600')
window.resizable(width=False, height=False)

builtins.window = window

# create all of the main containers
top_frame = tk.Frame(window, bg='red')
mid_frame = tk.Frame(window)
bottom_frame = tk.Frame(window, bg='yellow')
centreL = tk.Frame(mid_frame, bg='blue')
centreR = tk.Frame(mid_frame, bg='pink')

top_frame.grid(row=0, sticky="nsew")
mid_frame.grid(row=1, sticky="nsew")
bottom_frame.grid(row=2, sticky="nsew")
centreL.grid(row=0,column = 0, sticky = "nsew")
centreR.grid(row=0,column = 1, sticky = "nsew")

# layout all of the main containers
window.grid_columnconfigure(0, weight=1)
window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(1, weight=9)
window.grid_rowconfigure(2, weight=1)

mid_frame.grid_rowconfigure(0,weight=1)
mid_frame.grid_columnconfigure(0,weight=1)
mid_frame.grid_columnconfigure(1,weight=1)

centreL.grid_rowconfigure(0,weight=1)
centreL.grid_rowconfigure(2,weight=1)
centreL.grid_columnconfigure(0,weight=1)
centreL.grid_columnconfigure(2,weight=1)

centreR.grid_rowconfigure(0,weight=0)
centreR.grid_columnconfigure(0,weight=1)
centreR.grid_rowconfigure(1,weight=0)
centreR.grid_columnconfigure(2,weight=1)

for i in range(10):
    bottom_frame.grid_columnconfigure(i,weight = 1)

#placing 
#head label
head_L = tk.Label(top_frame,text = 'Example of Interface',font=(None,20),height =1)
head_L.grid(row=0,column = 0,sticky = 'ew',padx = 20,pady = 10)

#check boxes
willshowvdo = tk.BooleanVar()
tk.Checkbutton(centreR, text="Show vdo", variable=willshowvdo).grid(row=1,column = 1, pady = 10)

#text_widget output
text_widget = tk.Text(centreL,width=80,height=35)
text_widget.grid(row = 1,column = 1,padx = 5,pady = 5,sticky = 'ns')
#text_widget.config(state=tk.DISABLED)
builtins.text_widget = text_widget

#command button

browse_btn = tk.Button(bottom_frame, text ='Browse File(s)', command = lambda:open_file() ,activebackground = 'gray')
browse_btn.grid(row = 0,column = 7,pady = 10,sticky = 'ew')
brfol_btn = tk.Button(bottom_frame, text ='Browse Folder', command = lambda:open_folder() ,activebackground = 'gray')
brfol_btn.grid(row = 0,column = 8,pady = 10,sticky = 'ew')
st_btn = tk.Button(bottom_frame, text ='Proceed', command = lambda:proceed(willshowvdo) ,activebackground = 'gray')
st_btn.grid(row = 0,column = 9,padx = 10,pady = 10,sticky = 'ew')


#Running
window.mainloop()
