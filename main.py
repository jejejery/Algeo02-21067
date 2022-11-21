import tkinter
import tkinter.messagebox
import customtkinter
from PIL import ImageTk, Image
from tkinter import filedialog as fd
import cv2 
from backend.calculate import *
import zipfile
import time as t

customtkinter.set_appearance_mode("System")  
customtkinter.set_default_color_theme("blue") 


class App(customtkinter.CTk):

    WIDTH = 1500
    HEIGHT = 750

    def __init__(self):
        super().__init__()
        #DEKLARASI VARIABEL
        self.title("Face Recognition using Eigen Faces")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  
        self.testImage = None
        self.cap = None
        self.dataset_display = []
        self.dataset_train = np.array([])
        self.train_label = []

        self.imageTestGrayscale = np.ndarray(shape=(256*256))
        self.cv2image = None
        self.t_camera = 0
        self.state = False

        #TRAINING MODELS
        self.eigenfaces = None
        self.weight_training = None
        self.avg_training = None
        

        #METRICS
        self.euclidian_distance = 0
        self.cos_sim = 0
        self.label = ''


        # Terdiri atas 2 frame

        # Konfigurasi Awal Frame kanan dan kiri
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.imageUNDEF = ImageTk.PhotoImage((Image.open("no-image.png")).resize((512, 512), Image.ANTIALIAS))
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # Konfigurasi frame kiri

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, weight=1)   # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(4, weight=0)  # empty row as spacing
        self.frame_left.grid_rowconfigure(5, weight=0)  # empty row as spacing
        self.frame_left.grid_rowconfigure(6, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Face Recognition\nusing Eigen Face Algorithm",
                                              text_font=("Roboto Medium", -16))  # font name and size in px
        self.label_1.grid(row=3, column=0, pady=10, padx=10)

        self.button_1 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Input From Files",
                                                text_color="white",
                                                command=self.movetoFiles)
        self.button_1.grid(row=4, column=0, pady=10, padx=20)

        self.button_2 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Using Camera",
                                                text_color="white",
                                                command=self.movetoCamera)
        self.button_2.grid(row=5, column=0, pady=10, padx=20)

        self.label_mode = customtkinter.CTkLabel(master=self.frame_left, text="Appearance Mode:")
        self.label_mode.grid(row=9, column=0, pady=0, padx=20, sticky="w")

        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                        values=["Light", "Dark", "System"],
                                                        command=self.change_appearance_mode)
        self.optionmenu_1.grid(row=10, column=0, pady=10, padx=20, sticky="w")

        # ============ frame_right ============

        # Layour Sisi Kanan
        self.frame_right.columnconfigure((1, 2), weight=2)
        self.frame_right.columnconfigure(0, weight=1)
        self.frame_right.rowconfigure(0, weight=1)
        self.frame_right.rowconfigure(1, weight=2)
        self.frame_right.rowconfigure(2, weight=1)


        self.frame_input = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_input.grid(row=0, column=0, rowspan= 3, pady=20, padx=20, sticky="nsew")

        self.frame_info = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_info.grid(row=1, column=1,  pady=20, padx=20, sticky="nsew")

        self.frame_info2 = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_info2.grid(row=1, column=2,  pady=20, padx=20, sticky="nsew")

        # ============ frame_info ============

        # Layout Gambar
        self.frame_info.rowconfigure(0, weight=0)
        self.frame_info.rowconfigure(1, weight=1)
        self.frame_info.columnconfigure(0, weight=1)

        self.label_infot1 = customtkinter.CTkLabel(master=self.frame_info,
                                              text="Your Test Image",
                                              text_font=("Roboto Medium", -28))  
        self.label_infot1.grid(row=0, column=0, pady=20, padx=10)


        self.label_info_1 = customtkinter.CTkLabel(master=self.frame_info,
                                                   image = self.imageUNDEF,
                                                   corner_radius=6,
                                                   fg_color=("white", "gray38"),
                                                   justify=tkinter.LEFT)
        self.label_info_1.grid(column=0, row=1, sticky="nsew", padx=15, pady=15)

        self.frame_info2.rowconfigure(0, weight=0)
        self.frame_info2.rowconfigure(1, weight=1)
        self.frame_info2.columnconfigure(0, weight=1)

        self.label_infot2 = customtkinter.CTkLabel(master=self.frame_info2,
                                              text="Closest Result",
                                              text_font=("Roboto Medium", -28))  
        self.label_infot2.grid(row=0, column=0, pady=20, padx=10)

        self.label_info_2 = customtkinter.CTkLabel(master=self.frame_info2,
                                                   image = self.imageUNDEF,
                                                   corner_radius=6,
                                                   fg_color=("white", "gray38"),
                                                   justify=tkinter.LEFT)
        self.label_info_2.grid(column=0, row=1, sticky="nsew", padx=15, pady=15)

        self.label_infot3 = customtkinter.CTkLabel(master=self.frame_right,
                                              text="Execution time: ",
                                              text_font=("Roboto Medium", -20))  
        self.label_infot3.grid(row=2, column=1, columnspan=2, pady=20, padx=20, sticky="nw")

        # Layout Input
        self.frame_input.columnconfigure(0, weight=1)
        self.frame_input.rowconfigure(0, weight=1)
        self.frame_input.rowconfigure(4, weight=1)
        self.frame_input.rowconfigure(8, weight=1)
        self.frame_input.rowconfigure(10, weight=1)
        self.frame_input.rowconfigure(13, weight=3)
        self.label_radio_group = customtkinter.CTkLabel(master=self.frame_input,
                                                        text="Input Your Photo",
                                                        text_font=("Roboto Medium", -20))
        self.label_radio_group.grid(row=0, column=0, columnspan=1, pady=20, padx=10, sticky="")

        self.label_1 = customtkinter.CTkLabel(master=self.frame_input,
                                              text="Input Dataset",
                                              text_font=("Roboto Medium", -16))  
        self.label_1.grid(row=1, column=0, pady=0, padx=10)

        self.button_5 = customtkinter.CTkButton(master=self.frame_input,
                                                text="Choose File(s)",
                                                border_width=2,  
                                                text_color="white",
                                                fg_color="#1f6aa5",  
                                                state="normal",
                                                command=self.openDataset)
        self.button_5.grid(row=2, column=0, columnspan=1, pady=20, padx=20, sticky="we")

        self.label_1i = customtkinter.CTkLabel(master=self.frame_input,
                                              text="No Dataset(s) Found",
                                              text_font=("Roboto Medium", -10))  
        self.label_1i.grid(row=3, column=0, pady=0, padx=10)

        self.label_2 = customtkinter.CTkLabel(master=self.frame_input,
                                              text="Input Test File",
                                              text_font=("Roboto Medium", -16))  # font name and size in px
        self.label_2.grid(row=5, column=0, pady=0, padx=10)

        self.button_6 = customtkinter.CTkButton(master=self.frame_input,
                                                text="Choose File",
                                                state="normal",
                                                text_color="white",
                                                border_width=2,  # <- custom border_width
                                                fg_color="#1f6aa5",  # <- no fg_color
                                                command=self.openTestImage)
        self.button_6.grid(row=6, column=0, pady=20, padx=20, sticky="we")

        self.label_2i = customtkinter.CTkLabel(master=self.frame_input,
                                              text="No Test File Found",
                                              text_font=("Roboto Medium", -10))  
        self.label_2i.grid(row=7, column=0, pady=0, padx=10)

        self.button_7 = customtkinter.CTkButton(master=self.frame_input,
                                                text="START",
                                                state="normal",
                                                text_color="white",
                                                border_width=2,  # <- custom border_width
                                                fg_color="#1f6aa5",  # <- no fg_color
                                                command=self.button_event)
        self.button_7.grid(row=9, column=0, pady=20, padx=20, sticky="we")

        self.label_3 = customtkinter.CTkLabel(master=self.frame_input,
                                              text="Result",
                                              text_font=("Roboto Medium", -16))  # font name and size in px
        self.label_3.grid(row=11, column=0, pady=0, padx=10)
        
        self.label_4 = customtkinter.CTkLabel(master=self.frame_input,
                                              text="PRESS START",
                                              text_font=("Roboto Medium", -8))  # font name and size in px
        self.label_4.grid(row=12, column=0, pady=0, padx=5)

        self.label_5 = customtkinter.CTkLabel(master=self.frame_input,
                                              text="",
                                              text_font=("Roboto Medium", -8))  # font name and size in px
        self.label_5.grid(row=13, column=0, pady=0, padx=5)


        # set default values
        self.optionmenu_1.set("Dark")

    def button_event(self):
        tic = t.time()
        self.euclidian_distance, self.cos_sim, self.label = metrics_calculation(self.imageTestGrayscale,self.eigenfaces,self.weight_training,self.avg_training,self.train_label)

        ctr = 0
        for k in self.train_label:
            if k == self.label:
                break
            ctr += 1
        toc = t.time()
        
        ext_time ="Execution time: " + str(toc-tic) + "s"
        euclidian_distance_str = "Euclidian distance: " + "{0:.4E}".format(self.euclidian_distance)#str(euclidian_distance)
        cos_sim_str = "Cosine Similiarity: " + str(self.cos_sim)

        self.label_info_2.configure(image=self.dataset_display[ctr])
        self.label_4.configure(text = euclidian_distance_str)
        self.label_5.configure(text = cos_sim_str)
        self.label_infot3.configure(text = ext_time)
        
        

    
    def button_force(self):
        self.state = True

        
        
            
        

        
        
    def movetoFiles(self):
        self.label_radio_group.configure(text="Input Your Files")
        self.button_5.configure(state="normal")
        self.button_6.configure(state="normal")
        self.button_7.configure(state="normal")
        self.button_5.configure(fg_color="#1f6aa5")
        self.button_6.configure(fg_color="#1f6aa5")
        self.button_7.configure(fg_color="#1f6aa5")
        self.button_7.configure(command=self.button_event)
        if self.cap != None:
            self.cap.release()
            self.cap = None
        self.label_info_1.configure(image=self.imageUNDEF)
        self.label_infot1.configure(text="Your Test Image")
        self.state = False

    
    def movetoCamera(self):
        self.label_radio_group.configure(text="Look at Camera")
        self.label_infot1.configure(text="Camera Preview")
        self.button_5.configure(state="normal")
        self.button_6.configure(state="disabled")
        self.button_7.configure(state="normal")
        self.button_5.configure(fg_color="#1f6aa5")
        self.button_6.configure(fg_color=None)
        self.button_7.configure(fg_color="#1f6aa5")
        self.button_7.configure(command=self.button_force)
        self.label_info_1.configure(image=self.imageUNDEF)
        

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.t_camera = t.time()

        def show_frame_0():
            _, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            self.cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(self.cv2image)
            imgw, imgh = img.size
            imgtk = ImageTk.PhotoImage(image=img.crop([imgw/2-256, imgh/2-256, imgw/2+256, imgh/2+256]).resize((512,512)))
            self.label_info_1.imgtk = imgtk
            self.label_info_1.configure(image=imgtk)
            img_recognizing()
            self.label_info_1.after(10, show_frame_0)
            
        

        def img_recognizing():
            if(t.time()-self.t_camera > 0.5 and self.state):
               img_cropped = self.cv2image[:, 160:480,:]
               self.imageTestGrayscale = norm_img(img_cropped)
               self.euclidian_distance, self.cos_sim, self.label = metrics_calculation(self.imageTestGrayscale,self.eigenfaces,self.weight_training,self.avg_training,self.train_label)
               ctr = 0
               
               for k in self.train_label:
                   if k == self.label:
                       break
                   ctr += 1

               label_name ="Closest Result: " +  k
               euclidian_distance_str = "Euclidian distance: " + "{0:.4E}".format(self.euclidian_distance)#str(euclidian_distance)
               cos_sim_str = "Cosine Similiarity: " + str(self.cos_sim)
               self.label_info_2.configure(image=self.dataset_display[ctr])
               self.label_4.configure(text = euclidian_distance_str)
               self.label_5.configure(text = cos_sim_str)
               self.label_infot3.configure(text = label_name)
               self.t_camera = t.time()

        show_frame_0()
            
        
            



    def openTestImage(self):
        filetypes = (
            ('JPG Files', '*.jpg'),
            ('PNG Files', '*.png')
        )
        filename = fd.askopenfilename(
            title='Open Test Image',
            initialdir='/',
            filetypes=filetypes
        )
        
        if filename :
            self.imageTest = ImageTk.PhotoImage((Image.open(filename)).resize((512, 512), Image.ANTIALIAS))
            self.label_info_1.configure(image=self.imageTest)
            self.imageTestGrayscale = get_img_PIL(filename)
            filenameshow = ""
            lenname = len(filename)-1
            while (filename[lenname] != '/'):
                filenameshow = filename[lenname] + filenameshow
                lenname = lenname-1
            self.label_2i.configure(text=filenameshow)
    
    def openDataset(self):
        filetypes = (
            ('ZIP Files', '*.zip'),
        )
        
        self.dataset_display = []
        filename = fd.askopenfilename(
            title='Open Dataset',
            initialdir='/',
            filetypes=filetypes
        )
        

        if filename:
            filezip = zipfile.ZipFile(filename)
            listfile = filezip.namelist()
            self.dataset_train = np.ndarray(shape=(len(listfile),256*256))

            ctr = 0
            for itemfile in listfile:
                self.dataset_display.append(ImageTk.PhotoImage(Image.open(filezip.open(itemfile)).resize((512, 512), Image.ANTIALIAS)))
                x = filezip.open(itemfile)
                pic = get_img_PIL(x)
        
                self.dataset_train[ctr] = pic
                ctr += 1

            self.train_label = listfile

            # LAKUKAN TRAINING #
            self.training_dataset()
            # self.label_info_1.configure(image=self.dataset[0])
            # self.label_info_2.configure(image=self.dataset[1])
            # filenameshow = ""
            # lenname = len(filename)-1
            # while (filename[lenname] != '/'):
            #     filenameshow = filename[lenname] + filenameshow
            #     lenname = lenname-1

            self.label_1i.configure(text="Zip Loaded!")
            

    def training_dataset(self):
        tic = t.time()
        self.eigenfaces, self.weight_training, self.avg_training = training_parameters(self.dataset_train)
        toc = t.time()
        train_time ="Training time: " + str(toc-tic) + "s"
        self.label_infot3.configure(text = train_time)
        
        
    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def on_closing(self, event=0):
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()