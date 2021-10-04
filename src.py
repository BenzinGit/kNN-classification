# By: BenzinGit
import numpy as np
import math
import matplotlib.pyplot as plt 
import pandas as pd


# Method that generates training data
def generate_training_data (n_data_per_class, avg_std_matrix):
    
    n_classes  = len(avg_std_matrix) # Number of classes
    n_dimensions = ((len(avg_std_matrix[0,1:]))//2) # Number of dimensions
    n_columns = n_dimensions + 1 # Number of columns


    training_data = np.zeros([0,n_columns]) # Creates an empty matrix  
    
    # Two properties need to be generated per object
    if n_dimensions == 2:
       
        for a_class in range(n_classes): 
            
            for antal_data in range(n_data_per_class): 
                
               x = np.random.normal(avg_std_matrix[a_class,1], avg_std_matrix[a_class,2]) # Property 1 (x) generates 
               y = np.random.normal(avg_std_matrix[a_class,3], avg_std_matrix[a_class,4]) # Property 2 (y) generates
    
               training_data = np.vstack([training_data,[a_class+1, x, y]]) # The object gets added to training data matrix
        

    # Three properties need to be generated per object
    elif n_dimensions == 3:
        
        for a_class in range(n_classes): 
            
            for rader in range(n_data_per_class): 
                
               x = np.random.normal(avg_std_matrix[a_class,1], avg_std_matrix[a_class,2]) # Property 1 (x) generates 
               y = np.random.normal(avg_std_matrix[a_class,3], avg_std_matrix[a_class,4]) # Property 2 (y) generates 
               z = np.random.normal(avg_std_matrix[a_class,5], avg_std_matrix[a_class,6]) # Property 3 (z) generates 
            
               training_data = np.vstack([training_data,[a_class+1, x, y, z]]) # The object gets added to training data matrix
    
    training_data = np.round(training_data, 1) # Round to one decimal
    return training_data 

#________________________________________________________________________________________________________________________________________________


# Method that calculates and returns the expected classes of unkowns objects 
def kNN_algorithm(k, training_data, unknown_objects):
          
    n_training_data = len(training_data)
    
    classification_result = np.zeros([0,1]) # The classification results will be added to this matrix
    shortest_distance_results = np.zeros([0, k]) # The k shortest distances will be added to this matrix  

    
    # Compare and measure all objects with each other to find distance between the objects
    for unknown_object in unknown_objects:
            
            distance_per_object = np.zeros([0,2]) 
            
            for row in range(n_training_data): 
                known_object = training_data[row, 1:] # Known objects is retrieved from training_data
                distance = np.array(get_distance(known_object, unknown_object)) # Distance is measured
                a_class = np.array(training_data[row, 0]) # Gets the class the distance belongs to
                                
                distance_per_object = np.vstack([distance_per_object,[a_class, distance]]) # The data stacks in the matrix
                
                # Classifying each object
              
                # Sorts by distance to get the shortest distance to the unknown object
                distance_per_object = distance_per_object[distance_per_object[:, 1].argsort()] 
        
                # Slices to get only the k shortest values
                distance_per_object = distance_per_object[0:k]
        
                # Finds the class that occurs most often in distance_per_object, to classify the unknown object
                most_frequent_class = np.argmax(np.bincount(np.array(distance_per_object[:,0], 
                                                                      dtype="int64")))
                      
            # Stacking all the data in the matrixes  
            shortest_distance_results = np.vstack([shortest_distance_results,
                                                   [distance_per_object[:,1]]]) 
            classification_result = np.vstack([classification_result,
                                                 most_frequent_class])
            
    return classification_result, shortest_distance_results 
   
#________________________________________________________________________________________________________________________________________________

# Method that calculate the euclidean distance between two coordinates
def get_distance(point_1, point_2):
   
    n_dimensions = len(point_1) 
   
    # 2D
    if n_dimensions == 2: 
        distance = math.sqrt((point_1[0]-point_2[0])**2+ 
                            (point_1[1]-point_2[1])**2)    
    # 3D
    elif n_dimensions == 3: 
        distance = math.sqrt((point_1[0]-point_2[0])**2 + 
                            (point_1[1]-point_2[1])**2 + 
                            (point_1[2]-point_2[2])**2)    
        
    return np.round(distance, 3)

#________________________________________________________________________________________________________________________________________________
 

# Method to present the results in a scatter graph 
def present_results(k, training_data, unknown_objects, sorted_list, kNN_results, correct_classes = None):
    
 try: 
   n_classes = np.amax([training_data[:,0]]) 
   n_dimensions = len(training_data[0,1:]) 
   
    
   # Retrives important coordinates
   
   # Coortinates for class 1 traning objects 
   class1_x = training_data[training_data[:,0] == 1,1:][:,0] 
   class1_y = training_data[training_data[:,0] == 1,1:][:,1] 

   # Coortinates for class 2 traning objects 
   class2_x = training_data[training_data[:,0] == 2,1:][:,0] 
   class2_y = training_data[training_data[:,0] == 2,1:][:,1] 
   
   # Coortinates for class 3 traning objects 
   class3_x = training_data[training_data[:,0] == 3,1:][:,0] 
   class3_y = training_data[training_data[:,0] == 3,1:][:,1] 

   # Coortinates for unknown objects
   unknown_object_x = unknown_objects[:,0]
   unknown_object_y = unknown_objects[:,1]


   # Will run during choice 2 from the meny
   if correct_classes is not None:
       
        # Compares the results of the kNN algorithm with the correct results to 
        # create masks to find misclassifications and correct classifications
        
        mask_k1 = np.array((correct_classes[0,:] == kNN_results[:,0]) & # Class 1 correct classifications. 
                              (correct_classes[0,:] == 1))  
        mask_fk1 = np.array((correct_classes[0,:] != kNN_results[:,0]) & # Class 1 misclassifications.
                              (correct_classes[0,:] == 1))
        
        
        mask_k2 = np.array((correct_classes[0,:] == kNN_results[:,0]) & # Class 2 correct classifications. 
                              (correct_classes[0,:] == 2))
        mask_fk2 = np.array((correct_classes[0,:] != kNN_results[:,0]) & # Class 2 misclassifications. 
                              (correct_classes[0,:] == 2))
        
        
        mask_k3 = np.array((correct_classes[0,:] == kNN_results[:,0]) &  # Class 3 correct classifications. 
                              (correct_classes[0,:] == 3))
        mask_fk3 = np.array((correct_classes[0,:] != kNN_results[:,0]) & # Class 3 misclassifications.
                              (correct_classes[0,:] == 3))

    
        n_classifications = len(unknown_objects)    
        n_misclassifications = len(unknown_objects[mask_fk1]) + len(unknown_objects[mask_fk2]) + len(unknown_objects[mask_fk3])
        percent_misclassifications = (n_misclassifications / n_classifications)*100  
        
        # Writes the results in the console
        print("Current k-value:", k)
        print("Number of classifications performed:", n_classifications)
        print("Number of misclassifications:", n_misclassifications )
        print("Percentage misclassifications:", percent_misclassifications, "%" )


        # Creating a 2D graph
        if n_dimensions == 2:
                        
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_axes([0,0,1,1])
          
            ax.set_title("Graphic presentation of kNN classification")
            ax.set_xlabel("Property 1")
            ax.set_ylabel("Property 2")
                
            
            # Puts training objects into the graph
            plt.scatter(class1_x, class1_y,color = "r", label="Class 1") 
            plt.scatter(class2_x, class2_y,color = "b", label="Class 2") 
            
            if n_classes == 3:  
                plt.scatter(class3_x, class3_y,color = "g", label="Class 3") 
            
            # Adds every unknown object that got classified as class 1 
            plt.scatter(unknown_object_x[mask_k1], unknown_object_y[mask_k1],color = "orange", marker="s", label="Unknown class 1") 
            plt.scatter(unknown_object_x[mask_fk1], unknown_object_y[mask_fk1],color = "orange", marker="x", label="Misclassified")
    
            # Adds every unknown object that got classified as class 2 
            plt.scatter(unknown_object_x[mask_k2], unknown_object_y[mask_k2] ,color = "purple",marker="s", label="Unknown class 2") 
            plt.scatter(unknown_object_x[mask_fk2], unknown_object_y[mask_fk2] ,color = "purple", marker="x", label="Misclassified")
            
            
        
            # Adds every unknown object that got classified as class 3 (if number of classes = 3)  
            if n_classes == 3:  
                plt.scatter(unknown_object_x[mask_k3], unknown_object_y[mask_k3] ,color = "darkgreen",marker="s", label="Unknown class 3")
                plt.scatter(unknown_object_x[mask_fk3], unknown_object_y[mask_fk3] ,color = "darkgreen", marker="x", label="Misclassified")


        # Creating a 3D graph
        if n_dimensions == 3: 
           
           # Retrives Z coordinates for class 1, class 2, class 3 and unknown object
           class1_z = training_data[training_data[:,0] == 1,1:][:,2] 
           class2_z = training_data[training_data[:,0] == 2,1:][:,2] 
           class3_z = training_data[training_data[:,0] == 3,1:][:,2] 
           unknown_object_z = unknown_objects[:,2] 

    
      
           fig = plt.figure(figsize=(10,10))
           ax = plt.axes(projection="3d")
          
           # Puts training objects into the graph
           ax.scatter(class1_x, class1_y, class1_z, color="r", label="Class 1" ) 
           ax.scatter(class2_x, class2_y, class2_z, color="b", label="Class 2" ) 
           if n_classes == 3: 
               ax.scatter(class3_x, class3_y, class3_z, color="g", label="Class 3")
           
            
           # Adds every unknown object that got classified as class 1 
           ax.scatter(unknown_object_x[mask_k1],
                      unknown_object_y[mask_k1], 
                      unknown_object_z[mask_k1], label="Unknown class 1", color="orange", marker="s")
           
           ax.scatter(unknown_object_x[mask_fk1],
                      unknown_object_y[mask_fk1], 
                      unknown_object_z[mask_fk1], label="Misclassified", color="orange", marker="x")
           
           # Adds every unknown object that got classified as class 2
           ax.scatter(unknown_object_x[mask_k2],  
                      unknown_object_y[mask_k2], 
                      unknown_object_z[mask_k2], label="Unknown class 2", color="purple", marker="s")
           
           ax.scatter(unknown_object_x[mask_fk2], 
                      unknown_object_y[mask_fk2], 
                      unknown_object_z[mask_fk2], label="Misclassified", color="purple", marker="x")
           
           # Adds every unknown object that got classified as class 3 (if number of classes = 3) 
           if n_classes == 3: 
               ax.scatter(unknown_object_x[mask_k3], 
                          unknown_object_y[mask_k3], 
                          unknown_object_z[mask_k3], label="Unknown class 3", color="darkgreen", marker="s")
               
               ax.scatter(unknown_object_x[mask_fk3],
                          unknown_object_y[mask_fk3], 
                          unknown_object_z[mask_fk3], label="Misclassified", color="darkgreen", marker="x")


           ax.set_title("Graphic presentation of kNN classification")
           ax.set_xlabel("Property 1")
           ax.set_ylabel("Property 2")
           ax.set_zlabel("Property 3")
           

   #If correct_classes == None (during choice 1 from the menu)
   else:
        fig = plt.figure(figsize=(8,8))
        
        ax = fig.add_axes([0,0,1,1])
        ax.set_title("Graphic presentation of kNN classification")
        ax.set_xlabel("Property 1")
        ax.set_ylabel("Property 2")
        plt.axis('equal')
    
        plt.scatter(class1_x, class1_y,color = "r", label="Class 1" ) 
        plt.scatter(class2_x, class2_y,color = "b", label="Class 2" ) 
        plt.scatter(unknown_object_x, unknown_object_y ,color = "y", label="Unknown object") 
        
        # Creates a radius with K known elements inside
        longest_radius = sorted_list[:,-1] 
        radius = plt.Circle((unknown_object_x, unknown_object_y), longest_radius, fill=False) 
        ax.add_artist(radius) 

        print("Current k-value:", k)
        print("The algorithm classified the unknown object into class ",kNN_results[0,0])
        
   ax.legend()
   plt.show()
   return True # If plotting succeded returns true
 
 except: 
     return False  # If plotting failed it will return false




#________________________________________________________________________________________________________________________________________________


# Method that will show a menu and ask user to choose an option
def show_menu():
 
  while(True):
          print("-------------------------------------------------------------------")
          print("Select one of the following options:")
          print("1. Generate data and classify an unknown object")
          print("2. Measure the accuracy of the algorithm through CSV files")
          print("3. Exit program")
       
          choice = input("Enter menu options: ")  

          if (choice == "1"):
              try:
     
                    k = int(input("Enter K-value: ")) 
                    if k < 1: 
                        raise Exception("The K value cannot be less than 1. Please try again!\n") 
                    
                    n_data_per_class = int(input("Enter the number of training data per class: "))
                    if n_data_per_class < 1:
                        raise Exception("Number of data per class cannot be less than 1. Please try again!\n")     
                    if k > (n_data_per_class*2): 
                        raise Exception ("The k-value cannot be greater than the number of training data. Please try again!\n")     
                    
                    mv_e1_1 =  float(input("Class 1: Mean value for property 1: "))
                    std_e1_1 = float(input("Class 1: Standard deviation for property 1: "))
                    if(std_e1_1 < 0): 
                        raise Exception("The standard deviation cannot be less than 0. Please try again!\n")
                    
                    mv_e2_1 =  float(input("Class 1: Mean value for property 2: "))
                    std_e2_1 = float(input("Klass 1: Standard deviation for property 2: ")) 
                    if(std_e2_1 < 0): 
                        raise Exception("The standard deviation cannot be less than 0. Please try again\n!")
                  
                    mv_e1_2 =  float(input("Class 2: Mean value for property 1: "))
                    std_e1_2 = float(input("Klass 2: Standard deviation for property 1: "))
                    if(std_e1_2 < 0):
                        raise Exception("The standard deviation cannot be less than 0. Please try again\n")
                    
                    mv_e2_2 =  float(input("Class 2: Mean value for property 2: "))
                    std_e2_2 = float(input("Class 2: Standard deviation for  2: "))
                    if(std_e2_2 < 0):
                        raise Exception("The standard deviation cannot be less than 0. Please try again!\n")
    
                    
                    unknown_X = float(input("Enter the X coordinate of the unknown object: "))
                    unknown_Y = float(input("Enter the Y coordinate of the unknown object: "))
                    unknown_object = np.array([[unknown_X, unknown_Y]])
                   
                
              except ValueError: print("Value has to be a number.\n") 
              except Exception as e: print(e) 
              
              
              else:  
                  
                  mean_std_matris = np.array([[1,mv_e1_1, std_e1_1, mv_e2_1, std_e2_1 ], 
                                               [2,mv_e1_2, std_e1_2, mv_e2_2, std_e2_2 ]], dtype="int64")
                  try:
                      training_data = generate_training_data(n_data_per_class, mean_std_matris) 
                      kNN_results, sorted_list = kNN_algorithm(k, training_data, unknown_object) 
                      is_successful = present_results(k, training_data,unknown_object, sorted_list, kNN_results, None) 
                      
                      if (is_successful == False): 
                          print("Plotting failed.\n")
                  except: 
                     print("Calculations failed.\n") 


          elif(choice == "2"):
              try:
                  k = int(input("Enter k-value: ")) 
                  if k < 1: raise Exception("K value can't be less than 1. Try again!\n")
                   
                  filename = input("Enter the file name that contains the training data: ") 
                  training_data = pd.read_csv("data_files/"+filename+".csv", sep = ",", header=None) 
                  training_data = training_data.values 
                    
                  filename = input("Enter the file name that contains the unknown objects: ") 
                  unknown_objects = pd.read_csv("data_files/"+filename+".csv", sep = ",", header=None) 
                  unknown_objects = unknown_objects.values 
        
        
                  filename = input("Enter the file name that contains the class affiliation of the unknown objects: ") 
                  correct_classification = pd.read_csv("data_files/"+filename+".csv", sep = ",", header=None) 
                  correct_classification = correct_classification.values 

              except ValueError: print("The value must be a positive integer. The file may also be corrupt\n")
              except FileNotFoundError: print("Following file not found: ", filename+".csv") 
              except Exception as e: print(e) 
              
              
              else: 
                  try:           
                      kNN_results, sorted_list = kNN_algorithm(k, training_data, unknown_objects)  
                      is_successful = present_results(k, training_data, unknown_objects, sorted_list, kNN_results, correct_classification)
                      if (is_successful == False):
                          print("Plotting failed.\n")
                  except: 
                      print("The calculation failed. The files may be corrupt or the k-value exceeds the number of training data.\n")
                
        
          elif(choice == "3"): 
              print("Exit")
              return 0      
        
          else: print("Invalid selection. Please enter a value between 1-3\n") 
                
#________________________________________________________________________________________________________________________________________________
      



# Starts the program
show_menu() 


