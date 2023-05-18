import io
import tkinter
from tensorflow.keras.models import load_model
import zipfile
import os
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt



def open_main():
    img_window.destroy()
    root.deiconify()


model_path = 'mri_model.h5'
model = load_model(model_path)


num_slices=5

class_labels = ['pd']



def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))


def load_and_preprocess_images_from_zip(zip_path, folder_name):
    images = []
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        # Assuming the images in the zip file are named as image1.png, image2.png, ..., image5.png
        for i in range(1, 6):
            image_path = f'{folder_name}/mid_{i}_slice.png'
            with zip_file.open(image_path) as file:
                image = Image.open(file)
                # Resize the image to match the input size of the model
                image = image.resize((256, 256))
                image = image.convert('RGB')  # Convert the image to RGB format
                # Normalize pixel values to the range [0, 1]
                image = np.array(image) / 255.0
                images.append(image)
    images = np.array(images)
    return images


def test_model_from_zip(zip_path, folder_name):
    images = load_and_preprocess_images_from_zip(zip_path, folder_name)
    predictions = model.predict(images)
    conf_avg = np.mean(np.max(predictions, axis=1) * 100)
    return conf_avg


def load_mri():
    global img, img_window
    file_path = filedialog.askopenfilename(
        filetypes=[("Niffi", "*.gz")])
    
    if file_path.endswith(".nii.gz"):

        img = nib.load(file_path)
        img_data = img.get_fdata()

        output_folder_name = 'temp_slices'
        output_path = os.path.join(
             output_folder_name)
        os.makedirs(output_path, exist_ok=True)

        mid_slice = img_data.shape[2] // 2

        for i in range(-(num_slices//2), (num_slices//2)if (num_slices % 2 == 0) else (num_slices//2)+1):

            slice_data = np.squeeze(img_data[:, :, mid_slice+i])
            slice_data = (slice_data - np.min(slice_data)) / \
                (np.max(slice_data) - np.min(slice_data)) * 255  # normalize
            slice_data = slice_data.astype(np.uint8)

            output_filename = os.path.join(
                output_path, 'mid_'+str(i+(num_slices//2)+1)+'_slice.png')
            plt.imsave(output_filename, slice_data, cmap="gray")

            zipf = zipfile.ZipFile('temp_slices.zip', 'w', zipfile.ZIP_DEFLATED)


            zipdir('temp_slices', zipf)
            zipf.close()

            folder_name = 'temp_slices'
            folder_name = folder_name[:21]

            with open('temp_slices.zip', 'rb') as f:
                zipf_bytes = io.BytesIO(f.read())


            result_str = test_model_from_zip(zipf_bytes, folder_name)
            # result_label = f'The confidence value is {result_str:.2f}.'
            print(f"{result_str:.2f}% confidance")
            if result_str > 50:
                result_label = ' The person may have Parkinson\'s disease.'

            else:
                result_label = ' The person may not have Parkinson\'s disease.'




    # Hide the main window
    root.withdraw()

    # Open a new window to show the image

    img = Image.open('temp_slices/mid_3_slice.png')


    photo = ImageTk.PhotoImage(img)

    img_window = Toplevel(root)
    img_window.geometry("500x300")
    label = Label(img_window, image=photo)


# Keep a reference to the image object to prevent it from being garbage collected
    label.image = photo
    label.pack()
    text = Label(img_window, text=result_label)
    text.pack()
    back_button = Button(img_window, text="Back", command=open_main)
    back_button.pack()


# Create the main application window
root = Tk()
root.geometry("500x300")

# Create a button that will open the file dialog
button = Button(root, text="Load MRI", command=load_mri)
button.grid(row=0, column=0, sticky=NSEW)

# Configure grid weights to center the button
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
button.pack()

root.title("PD Detection")


# Start the application
root.mainloop()
