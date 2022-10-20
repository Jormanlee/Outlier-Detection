##Editor Qiuming Li
import numpy as np
import cv2 as cv
import glob



def get_data():
    
    ##reads Data from directory 'images'
    ##Output: numpy.arrays for Input_rgb = rgb images, Input_z = bad depth images, Output = good depth images
    
    Input_rgb = []
    Input_z = []
    Output = []
    counter = 1
    for file_name in glob.glob(r"C:/Users/49162/Desktop/python/images"+'/Camera*'):
        for depth_file in glob.glob(r"C:/Users/49162/Desktop/python/images" + '/depth' + str(counter) + '_*'):
            if not depth_file[-5:] == '1.png' :
                Input_rgb.append(cv.imread(file_name))
                Input_z.append(cv.imread(depth_file))
                Output.append(cv.imread(depth_file[:-5]+ '1.png'))
        counter += 1
    return np.array(Input_rgb), np.array(Input_z), np.array(Output)


def get_dataset(train_size=0.8, val_size=0.1, test_size=0.1)##shuffle the dataset
    ## A method to transform blender data into shuffled and splitted data  
    ##train_size: percentage, hwo much data will be used for training [float]
    ##test_size: percentage, hwo much data will be used for testing [float]
    ##val_size: percentage, how much data will be used for validation [float]
    
    input_rgb, input_z, output = get_data() 
    input_rgb_train, input_rgb_val, input_rgb_test, input_z_train, input_z_val, input_z_test, output_train, output_val, output_test = _split_data(input_rgb, input_z, output, train_size, test_size, val_size)

    return input_rgb_train, input_rgb_val, input_rgb_test, input_z_train, input_z_val, input_z_test, output_train, output_val, output_test
    ##*_train: data for training [list]
    ##*_test: data for testing [list]
    ##*_val: data vor validation [list]
 
 def get_dataset_matrix(train_size=0.8, test_size=0.1, val_size=0.1):# to change in 4 dimensions r g b z
    ##This method concatenates the input rgb values with the input_z values. 
    ##For this, Input_z has to be upsampled to a size of 256x256 , because our z images are 128*128
    ##Input = train_-, test_-, val_size in percent/100, 0.8 means 80%
    

    input_rgb_train, input_rgb_val, input_rgb_test, input_z_train, input_z_val, input_z_test, output_train, output_val, output_test = get_dataset(train_size, val_size, test_size)
    input_train = []
    input_val = []
    input_test = []
    for i in range(len(output_train)):
        input_train.append(np.concatenate((input_rgb_train[i],cv.resize(input_z_train[i],(256,256), interpolation=cv.INTER_AREA)[:,:,0:1]), axis = 2)) #adds 4th dimension to input_rgb values [r, g, b, z_value] for each pixel
    for i in range(len(output_test)):
        input_test.append(np.concatenate((input_rgb_test[i],cv.resize(input_z_test[i], (256,256), interpolation=cv.INTER_AREA)[:,:,0:1]),axis = 2)) #adds 4th dimension to input_rgb values [r, g, b, z_value] for each pixel
    for i in range(len(output_val)):
        input_val.append(np.concatenate((input_rgb_val[i],cv.resize(input_z_val[i], (256,256),interpolation=cv.INTER_AREA)[:,:,0:1]),axis = 2)) #adds 4th dimension to input_rgb values [r, g, b, z_value] for each pixel

    return np.array(input_train), np.array(input_val), np.array(input_test), np.array(output_train), np.array(output_val), np.array(output_test)
    ##Output = Input_train, -_test, -_val  as [r,g,b,z]

def _split_data(Input_rgb, Input_z, Output, train_size = 0.8, test_size=0.1, val_size = 0.1):

    ##Parameter to split the data into several pieces. The Data will be shuffled
    ##Parameters:
    ##Input_rgb, Input_z, Output: list of numpy arrays of data [list]
    ##train_size: percentage, hwo much data will be used for training [float]
    ##test_size: percentage, hwo much data will be used for testing [float]
    ##val_sie: percentage, hwo much data will be used for validation [float]
    
    
    
    if (train_size + test_size + val_size) > 1.0:
        print("You can't split the values above 1.0 (=100%):\ntrain_size: {}; test_size: {}".format(train_size, test_size))
        return None, None, None
    
    #Calculate the length of the samples
    train_len = int(len(Output) * train_size)
    val_size = int(len(Output) * val_size)
    
    #Set the random seet
    np.random.seed(0)
    
    #Shuffling indexs
    ind = np.arange(len(Output))
    np.random.shuffle(ind)
    Input_rgb = [Input_rgb[i] for i in ind]
    Input_z = [Input_z[i] for i in ind]
    Output = [Output[i] for i in ind]
    
    #Split data
    Input_rgb_train = Input_rgb[:train_len]
    Input_z_train = Input_z[:train_len]
    Output_train = Output[:train_len]

    Input_rgb_val = Input_rgb[train_len:(train_len + val_size)]
    Input_z_val = Input_z[train_len:(train_len + val_size)]
    Output_val = Output[train_len:(train_len + val_size)]

    Input_rgb_test = Input_rgb[(train_len + val_size):]
    Input_z_test = Input_z[(train_len + val_size):]
    Output_test = Output[(train_len + val_size):]

    return Input_rgb_train, Input_rgb_val, Input_rgb_test, Input_z_train, Input_z_val, Input_z_test, Output_train, Output_val, Output_test


##transform it to torch value used for tensor
##train_data: list of training_data [list]
##test_data: list of test_data [list]
##val_data: list of validation_data [list]
def _transform_array_to_tensor(Input_rgb_train, Input_rgb_val, Input_rgb_test, Input_z_train, Input_z_val, Input_z_test, Output_train, Output_val, Output_test):
   
    train_data = tf.data.Dataset.from_tensor_slices((Input_rgb_train, Input_z_train, Output_train))
    val_data = tf.data.Dataset.from_tensor_slices((Input_rgb_val, Input_z_val, Output_val))
    test_data = tf.data.Dataset.from_tensor_slices((Input_rgb_test, Input_z_test, Output_test))
    #does not work
    
    return train_data, test_data, val_data


##Possible Method-Call
##files = ["output.csv","output.csv","output.csv","output.csv","output.csv","output.csv","output.csv","output.csv"]
##col_x_y = ["x_pixel", "y_pixel"]
##col_data = ["r", "g", "b"]
##col_pred = ["z"]

##get_csv_tf_dataset(files, col_x_y, col_data, col_pred, train_size = 0.5, test_size=0.25, val_size= 0.25)

#a,b,c,d,e,f =get_dataset_matrix(0.8,0.1,0.1)
#print("in : ", a, "out: ", d)







