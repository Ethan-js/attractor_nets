#This is the sample code of discrete hopfield network


import numpy as np
import random
from PIL import Image
import os
import re

#convert matrix to a vector
def mat2vec(x):
    m = x.shape[0]*x.shape[1]
    tmp1 = np.zeros(m)

    c = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp1[c] = x[i,j]
            c +=1
    return tmp1


#Create Weight matrix for a single image
def create_W(x):
    if len(x.shape) != 1:
        print("The input is not vector")
        return
    else:
        w = np.zeros([len(x),len(x)])
        for i in range(len(x)):
            for j in range(i,len(x)):
                if i == j:
                    w[i,j] = 0
                else:
                    # weight of connection between two nodes is +1 if same value (1*1 or -1 * -1),
                    # -1 if opposite value (very basic way to set up weights)
                    w[i,j] = x[i]*x[j]
                    w[j,i] = w[i,j]
    return w


#Read Image file and convert it to Numpy array
def readImg2array(file,size, threshold= 145):
    pilIN = Image.open(file).convert(mode="L")
    pilIN= pilIN.resize(size)
    #pilIN.thumbnail(size,Image.ANTIALIAS)
    imgArray = np.asarray(pilIN,dtype=np.uint8)
    x = np.zeros(imgArray.shape,dtype=np.float)
    x[imgArray > threshold] = 1
    x[x==0] = -1
    return x

#Convert Numpy array to Image file like Jpeg
def array2img(data, outFile = None):

    #data is 1 or -1 matrix
    y = np.zeros(data.shape,dtype=np.uint8)
    y[data==1] = 255
    y[data==-1] = 0
    img = Image.fromarray(y,mode="L")
    if outFile is not None:
        img.save(outFile)
    return img

def g(y_vec,w,sigma):
    # print(y_vec)
    # print(np.linalg.norm(y_vec-w) ** 2)
    return np.exp(-np.linalg.norm(y_vec-w) ** 2 / (2 * sigma ** 2))

def entropy(q):
    ent_sum = 0
    for i in range(len(q)):
        ent_sum += q[i] * np.log2(1/q[i])
    return ent_sum


#Update
def update(attractor_nodes,E,sigma,n,time=25,threshold=0.00001,mu=0.5):
    y_t = E
    sigma_t = sigma
    # Initialise alpha_t to 1
    alpha_t = 1
    # initialise activity vector q
    print(len(attractor_nodes))
    q = [0 for i in range(len(attractor_nodes))]
    for s in range(time):
        # find attractor sum
        cur_sum = 0
        for attractor in attractor_nodes:
            # if s <= 10:
                # print(-np.linalg.norm(y_t-attractor[0])**2 / (2 * sigma ** 2))
            cur_sum += attractor[1] * g(y_t,attractor[0],sigma_t)
        # determine activity of each attractor
        for i in range(len(q)):
            attractor = attractor_nodes[i]
            q[i] = attractor[1] * g(y_t,attractor[0],sigma_t) / cur_sum
        # find sum of activities
        act_sum = 0
        for i in range(len(q)):
            act_sum += q[i] * attractor_nodes[i][0]

        # determine the next state of the network
        # mu determines amount of context-independent memory)
        y_next = mu * y_t + (1 - mu) * (alpha_t * E  + (1 - alpha_t) * act_sum)
        # set alpha_t to 0 (for now)
        alpha_t = 0
        # update sigma_t
        sig_sum = 0
        for i in range(len(q)):
            sig_sum += q[i] * np.linalg.norm(y_t - attractor_nodes[i][0]) ** 2
        # for some reason, the sigma update does not work and goes to 0
        # actually, this makes sense if the model converges very quickly - maybe stop it earlier
        # if s <= 10:
        #     print(sig_sum/n)
        # sigma_t = np.sqrt(sig_sum / n)

        # Update the attractors!
        print([np.linalg.norm(y_t-entry[0]) for entry in attractor_nodes])
        print(min([entry[0] for entry in attractor_nodes],key=lambda x: np.linalg.norm(y_t-x)))
        for i in range(len(attractor_nodes)):
            nu = 1 / (2 * q[i])
            nu = 0
            attractor_nodes[i][0] = nu * q[i] * y_t + (1 - nu * q[i]) * attractor_nodes[i][0]
        if entropy(q) < threshold:
            return y_t
        # if s > 2 and np.linalg.norm(y_t-y_next) < threshold:
        #     return y_t
        y_t = y_next
    return y_t


#The following is training pipeline
#Initial setting
def hopfield(train_files,test_files,time=1000,
    sigma=10,size=(100,100),threshold=60, current_path=None):

    #read image and convert it to Numpy array
    print("Importing images and creating weight matrix....")

    #num_files is the number of files
    num_files = 0
    for path in train_files:
        print(path)
        x = readImg2array(file=path,size=size,threshold=threshold)
        array2img(x).show()
        x_vec = mat2vec(x)
        # print(len(x_vec))
        if num_files == 0:
            # initialise list of attractor nodes
            attractor_nodes = [[x_vec,1]]
            num_files = 1
        else:
            # add a (w_i,pi_i) pair for the new node
            attractor_nodes.append([x_vec,1])
            num_files +=1

    print("Attractor nodes are set")


    #Import test data
    counter = 0
    for path in test_files:
        y = readImg2array(file=path,size=size,threshold=threshold)
        oshape = y.shape
        y_img = array2img(y)
        # y_img.show()
        print("Imported test data")

        y_vec = mat2vec(y)
        print("Updating...")
        y_vec_after = update(attractor_nodes=attractor_nodes,
            E=y_vec,n=len(test_files),time=time,sigma=sigma)
        y_vec_after = y_vec_after.reshape(oshape)
        if current_path is not None:
            outfile = current_path+"/after_"+str(counter)+".jpeg"
            array2img(y_vec_after,outFile=outfile)
        else:
            after_img = array2img(y_vec_after,outFile=None)
            after_img.show()
        counter +=1


#Main
#First, you can create a list of input file path
current_path = os.getcwd()
train_paths = []
path = current_path+"/train_pics/"
path = current_path+"/general_pics/vanilla_boys/"
for i in os.listdir(path):
    if re.match(r'[0-9a-zA-Z-_]*.jp[e]*g',i):
        train_paths.append(path+i)

#Second, you can create a list of sunglasses file path
test_paths = []
path = current_path+"/test_pics/"
path = current_path+"/general_pics/the_boys/"
for i in os.listdir(path):
    if re.match(r'[0-9a-zA-Z-_]*.jp[e]*g',i):
        test_paths.append(path+i)

#Hopfield network starts!
hopfield(train_files=train_paths, test_files=test_paths, time=20000,size=(100,100),threshold=60, current_path = current_path)

