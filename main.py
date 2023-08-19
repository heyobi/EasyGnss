from tkinter import *
from tkinter import filedialog as fd
import customtkinter
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue") #"blue", "green", "dark-blue", "sweetkind"
from Python_codes_gps_calc import main_gps_calc as gps_calc_func

root = customtkinter.CTk()
root.geometry("300x700")
root.title("EasyGnss")



def select_file_sp3(entry_field):
    filetypes = (
        ('text files', '*.sp3'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    entry_field.delete(0,END)
    entry_field.insert(0,filename)
    return filename

def select_file_RINEX(entry_field):
    filetypes = (
        ('Observation File', '*.*o'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    entry_field.delete(0,END)
    entry_field.insert(0,filename)
    return filename



sp3_txt_path =customtkinter.CTkEntry(master=root,width= 240)
sp3_txt_path.place(relx=0.5,rely=0.1,anchor=CENTER)
#sp3_txt_path.pack(expand=True)

Sp3_path = None 
button_sp3_path = customtkinter.CTkButton(master=root, text="Select sp3 file", width=240, height=50, command=lambda: select_and_assign_sp3())

def select_and_assign_sp3():
    global Sp3_path
    Sp3_path = select_file_sp3(sp3_txt_path)
    print(Sp3_path)
    return Sp3_path


button_sp3_path.place(relx=0.5,rely=0.17,anchor=CENTER)
#button_sp3_path.pack(expand=True)

RINEX_txt_path =customtkinter.CTkEntry(master=root,width= 240)
RINEX_txt_path.place(relx=0.5,rely=0.3,anchor=CENTER)
#RINEX_txt_path.pack(expand=True)

Rinex_path = None
button_RINEX_path = customtkinter.CTkButton(master=root,text="Select RINEX file",width= 240,height=50,command=lambda: select_and_assign_RINEX())
def select_and_assign_RINEX():
    global Rinex_path
    Rinex_path = select_file_RINEX(RINEX_txt_path)
    print(Rinex_path)
    return Rinex_path

button_RINEX_path.place(relx=0.5,rely=0.37,anchor=CENTER)
#button_RINEX_path.pack(expand=True)

button_calculation = customtkinter.CTkButton(master=root,text="Calculate Position of Reciver",width= 240,height=50,command=lambda: gps_konum_yaz())
button_calculation.place(relx=0.5,rely=0.5,anchor=CENTER)
#button_calculation.pack(expand=True)
def gps_konum_yaz():
    X,Y,Z,PSD,PDOP = gps_calc_func(str(Rinex_path),str(Sp3_path))
    posx = customtkinter.CTkLabel(root, text=f"X: {X:.3f} m\n\nY: {Y:.3f} m\n\nZ: {Z:.3f} m\n\nPSD: {PSD:.3f}\n\nPDOP: {PDOP:.3f}",font=('Arial', 24))
    posx.place(relx=0.5,rely=0.75,anchor=CENTER)

root.mainloop()
