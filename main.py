import os
from preprocessing import preprocess_main
from model_VGG import train_vgg
from model_resnet50 import train_resnet
from model_inceptionV3 import train_inceptionv3
from model_efficientnet import train_efficientnet
from model_densenet import train_densenet
from post_processing import post_processing



def main(directory, target_name):
    try:
        # List all entries in the directory
        entries = os.listdir(directory)
        # Return only the folder that matches the target_name
        folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry)) and entry == target_name]
        new_directory_path = os.path.join(directory, target_name)
        print("Filtered folders in directory:", new_directory_path )
        
        # Call the preprocessing script
        if folders:
            #preprocess run
            # preprocess_main(new_directory_path, img_size=224)

            #train Vgg16
            # train_vgg(new_directory_path,directory)

            # #train resnet50
            # train_resnet(new_directory_path,directory)

            #train inceptionV3
            # train_inceptionv3(new_directory_path,directory)

            #train EfficientNetB0
            train_efficientnet(new_directory_path,directory)

            #train EfficientNetB0
            train_densenet(new_directory_path,directory)

            #post_processing
            post_processing(new_directory_path,directory)

        return folders
    except Exception as e:
        print(f"Error listing folders in {directory}: {e}")
        return []

# Example usage

if __name__ == "__main__":
    directory = os.getcwd() 
    target_name = 'data'
    folders = main(directory, target_name)

    print("done")


