from django.shortcuts import render, redirect

#from django.contrib.auth.forms import UserCreationForm 
from django.contrib import messages
from .forms import UserRegisterForm
import time
import tensorflow as tf

# local imports
from .utils import detection as dt
from .utils import directory as dy
from .utils import labeling as label

from django.views.decorators.csrf import csrf_exempt,csrf_protect


# Huawei imports
from obs import ObsClient
import io
import base64


def home(request):
    return render(request,'home.html')

def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data['username']
            messages.success(request,f'Usuario {username} fue creado exitosamente!')
            time.sleep(1)
            return redirect('/dashboard')
    else:
        form = UserRegisterForm()
    
    context = {'form':form}

    return render(request,'account/register.html', context)

def resetPass(request):
    return render(request,'account/resetPass.html')

def confirmationEmail(request):
    return render(request,'success.html')


#############################################################################
########################## Huawei Gallery ###################################
#############################################################################

# Create an instance of ObsClient.
obsClient = ObsClient(
 access_key_id='------', 
 secret_access_key='------', 
 server='------'
)

def getImage(bucket_name,object_name):
    # Initializing obs client to get the image
    resp = obsClient.getObject(bucketName=bucket_name,objectKey=object_name,loadStreamInMemory=True)

    if resp.status < 300:
        # return image as bytes datatype
        imagebyte = resp.body.buffer
        
    return imagebyte

def dashboard(request):

    monthMapping = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dic':'12'}

    if request.GET.get('Year') or request.GET.get('Month') or request.GET.get('Day'):

        date_filtered = str(request.GET.get('Year')) +'/'+ monthMapping[str(request.GET.get('Month'))] +'/'+ str(request.GET.get('Day'))

        # Bucekt Name
        bucket_name = 'jaguars-dataset'

        # Initializing obs client to get all images as a bytes
        resp = obsClient.listObjects(bucketName=bucket_name, max_keys=12, prefix='dataset/val/jaguar_day')

        images_img = []
        images_key = []

        if resp.status < 300:
        
            index = 1
            for content in resp.body.contents:
                if index != 1:
                    date_img = content.lastModified.split(" ")[0] #yy/mm/dd
                    if date_filtered == date_img:
                        img = getImage(bucket_name=bucket_name,object_name=str(content.key))
                        images_img.append(base64.b64encode(img).decode())
                        images_key.append(content.key.split('/')[-1])
                        #content.lastModified
                        #content.etag
                        #content.size
                else:
                    index +=1
                index +=1

        # Close ObsClient
        obsClient.close()
        return render(request,'dashboard.html',{'images': images_img, 'keys':images_key})

    else:
        return render(request,'dashboard.html')


#############################################################################
################################ APIS! ######################################
#############################################################################
def demoModels(request):
    return render(request,'demo.html')

def objectDetection(request):
    return render(request,'mlModels/convolnet.html')

@csrf_exempt
def detection(request):
    #Allow files with extension png, jpg and jpeg
    valid_extensions = ['jpg' , 'jpeg' , 'png', 'zip', 'JPG','JPEG','PNG']

    new_model = tf.keras.models.load_model('model/object_detection_v1_23_02_2023.h5')

    file = request.FILES['imageFile']
    fileb = request.FILES['imageFile'].read()

    file_path = dy.save_image_static(file)

    img = dt.loadImage(fileb)

    outputs = [int(x[0][0]) for x in new_model.predict(img)]

    #pred= label.boundingbox(file_path,outputs)

    return render(request, 'mlModels/detect.html', {'outputs': outputs})
    