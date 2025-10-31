# Function Calling Script

try:
    from ultralytics import YOLO
except:
    import os
    os.system('pip uninstall ultralytics')
    os.system('pip install ultralytics')
    from ultralytics import YOLO

from PIL import Image
import os
import stat
import shutil
import pandas as pd


def get_latest_path(root_dir):
  files={filename.split('train')[-1]:filename for filename in os.listdir(root_dir) if filename.startswith("train")}

  return files[max(files.keys())]


def remove_readonly(func, path, exc_info):
    """
    Error handler for rmtree to handle read-only files.
    func: the function (os.remove or os.rmdir) that raised the error
    path: the path that caused the error
    exc_info: exception information
    """
    # make the file writable
    os.chmod(path, stat.S_IWRITE)
    # retry the function
    func(path)


def get_weights():
    """
    Returns the path to the pre-trained weights for the food classification model.
    """
    
    if not os.path.exists("./weights/best.pt"):
        weights_dir = "./weights"
        dst = os.path.join(weights_dir, "best.pt")
        os.makedirs(weights_dir, exist_ok=True)
        try:
            os.system('git clone https://huggingface.co/GboyeStack/NigerFoodAi ./model_clone')
        except:
            raise Exception("Unable to download model weights. Please check your internet connection.")
        else:
            shutil.copytree('./model_clone/runs','./runs')
            shutil.rmtree('./model_clone',onerror=remove_readonly)
            weights_path=f"./runs/classify/{get_latest_path('./runs/classify')}/weights/last.pt"
            shutil.copy(weights_path,dst)
            shutil.rmtree('./runs',onerror=remove_readonly)

    return "./weights/best.pt"


def load_model(model_path: str):
    """
    Loads the food classification model from the specified path.
    """
    
    model = YOLO(model_path)

    return model


def classify_food_image(image) -> str:
    """
    Classifies a food image and returns the predicted food name.
    """
    # model_path = get_weights()

    classifier = load_model(model_path="./weights/best.pt")

    predicted_food = classifier.names[classifier(image)[0].probs.top1]

    return predicted_food

def generate_response(info):

    #Will update the repsonse logic later

    pass

    return info.get('Description')


#To do: Update this logic 
def search_food_in_database(food_name: str, database: str) -> dict:
    """
    Searches for the food item in the nutritional database and returns its details.
    """
    #I have to update the logic in this function such that it doesn't only just search the Nigerian foods csv file but also searches all other sources
    #The goal is to ensure that the llm used has a many contenxt that is needed for generation.

    if database.lower() != 'nigerian foods':

        raise ValueError("Currently, only the 'Nigerian Foods' database is supported.")
    
    data = pd.read_csv('./data/Nigerian Foods.csv')

    # print(data.columns)

    food_data = data[data['Food_Name'].str.lower() == food_name.lower()]

    if not food_data.empty:

        return food_data.iloc[0].to_dict()
    
    else:
        return {}
    
def augment_output(food_name: str) -> str:
    """
    This functionality provides additional, context-rich details for each classified food item to improve user experience and make the responses more informative.

    """

    food_details= search_food_in_database(food_name, database='Nigerian Foods')

    if food_details:

        description= generate_response(food_details)

        return {'description':description,
                'origin': food_details.get('Region'," "),
                'spice_level':food_details.get('Spice_Level',' '),
                'Main_Ingredients':[ingredient for ingredient in food_details.get('Main_Ingredients').split(',') if ingredient]
                }

    else:
        return "No additional information found for the classified food item."
   


#Basically for testing purposes
if __name__ == "__main__":

    import os

    response=augment_output("Abacha")

    print(response)

