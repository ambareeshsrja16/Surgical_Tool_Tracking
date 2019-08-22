def get_DLC_prediction(file_path = None,
                       dlc_config_file = r"C:\Users\asree\Downloads\Tool_tracking_main\Tool_Tracking-Ambareesh-2019-07-31\config.yaml",
                       threshold=0.0,
                       display = False, marker_size = 10, save_path = None,  save_format='png'):
    """
    filepath: Complete file.path of the image to be tested
    threshold: The probability value for the prediction points. If a point is below threshold it won't be marked
    display: If True display the test and final images
    
    
    #TO DO:
    Remove boundary
    Preserve size as is
    
    """
    import pathlib
    import matplotlib.pyplot as plt
    import pandas as pd
    
    import deeplabcut
    path_config_file = dlc_config_file
    directory_path = str(pathlib.Path(file_path).parent)
    file_suffix = pathlib.Path(file_path).suffix
    deeplabcut.analyze_time_lapse_frames(path_config_file, directory=directory_path, frametype=file_suffix, save_as_csv=True) #looks promising
    im = plt.imread(file_path)[:,:,:3] #.PNG can have four channels!
    
    for file_name in pathlib.Path(file_path).parent.iterdir():
        if '.csv' in str(file_name):
            pred_csv = pd.read_csv(file_name, skiprows=[0,1], usecols=lambda x: x not in 'coords')
            X, Y = [], []
            for x, y, prob in zip(*[iter(pred_csv.columns)]*3):
                if pred_csv.loc[0,prob] > threshold:
                    X.append(pred_csv.loc[0,x])
                    Y.append(pred_csv.loc[0,y])
            
            plt.figure(figsize=(im.shape[0]/96, im.shape[1]/96)) # CHANGE TO INCORPORATE CORRECT SIZE
            implot = plt.imshow(im.copy())
            plt.scatter(x= X, y= Y, c= 'b', s=marker_size)
            if not save_path:
                plt.savefig('dlc_prediction.'+save_format, bbox_inches='tight', dpi = 'figure')
            else:
                plt.savefig(pathlib.Path(save_path).joinpath('dlc_prediction.'+save_format), bbox_inches='tight', dpi = 'figure')
            if not display:
                plt.close()
            print(f"DLC prediction done, file saved as dlc_prediction.{save_format}")
            break

if __name__ =='__main__':
    get_DLC_prediction(file_path=r"C:/Users/asree/Downloads/Tool_tracking_main/Garner/test_img.png", save_path= r"C:\Users\asree\Downloads\Tool_tracking_main\Garner")