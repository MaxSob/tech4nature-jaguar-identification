import os
import cv2

def boundingbox(file_path,outputs):

    directory='static/images/obj_det/' # CHECK\

    img = cv2.imread(file_path,1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    bounded =cv2.rectangle(img, (outputs[0], outputs[1]), (outputs[2], outputs[3]), (0,255,0), 4)
    
   
    filename = "saved.png"
    file_p = os.path.join('/jaguarweb/dashboard/static/images/obj_det/', filename) # CHECK
    
    
    cv2.imwrite(file_p, bounded)

    return directory+filename