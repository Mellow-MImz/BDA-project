import os
import glob
import random
import numpy as np
import cv2
import xml.etree.ElementTree as ET

text_file = open('/home/ten11/Desktop/send3/labels.txt', "r")
lines = text_file.read().split('\n')
classes = lines[:len(lines)-1]
# print(lines,'\n',classes)

#Takes in labels file name
def Name():
	p = os.listdir('./Annotation')
	labels = []
	labels1 = []
	#for i in range(len(p)):
	for i in range(5):
		labels1.append(p[i])
		s = p[i].split('-')
		#print(s)
		if len(s) >2:
			labels.append('_'.join(s[1:len(s)]))
			#print('im in ',labels[i])
		else:
			labels.append(s[1])
	#print(len(labels))
	with open('labels.names', 'w') as f:
		for item in labels:
		    f.write("%s\n" % item)
	with open('labels1.txt', 'w') as f:
		for item in labels1:
		    f.write("%s\n" % item)
	print('-------------------- Got labels --------------------')
	return 'labels.txt'

def moving(text,img_path, annots_path):
	text_file = open(text, "r")
	lines = text_file.read().split('\n')
	# print(len(lines))
	# print(lines)
	os.mkdir(os.getcwd()+'/Test_img')
	os.mkdir(os.getcwd()+'/Train_img')
	os.mkdir(os.getcwd()+'/Test_annots')
	os.mkdir(os.getcwd()+'/Train_annots')
	####### For every folder in the text file
	for i in range(len(lines)-1):
		path = img_path+lines[i]+"/*.jpg"
		Images = np.array(glob.glob(path))
		print(len(Images))
		length = len(Images)
		tr1 = random.sample(range(length), int(length*.7)+1)
		d = set(range(len(Images)))-set(tr1)
		te1 = list(d)
		# print(tr1,'\n',te1)
		x_test = Images[te1]
		x_train = Images[tr1]
		# print(len(x_test),len(x_train))
		# print(np.sort(te1),'\n' ,np.sort(tr1))
		newpath = os.getcwd()+'/Test_img/'
		newpath2 = os.getcwd()+'/Test_annots/'
		newpath3 = os.getcwd()+'/Train_img/'
		newpath4 = os.getcwd()+'/Train_annots/'

		for j in range(len(x_test)):
			m = x_test[j].split('/')[-1].split('.')[0]
			# print(annots_path+Images[j].split('/')[-1].split('.')[0])
			d = x_test[j].split('/')[-2].split('-')
			# print(d)
			filename = x_test[j].split('/')[-1].split('_')[1]
			if len(d)>2:
				folder = '_'.join(d[1:len(d)])
			else:
				folder = d[1]
			n = filename.split('.')[0]
			# print(x_test[j])
			# print(newpath+folder+'_'+filename)
			# print(annots_path+x_test[j].split('/')[-2]+'/'+m)
			# print(newpath2+folder+'_'+n+'.xml')
			# print(j,'\n')
			os.rename(x_test[j],newpath+folder+'_'+filename)
			os.rename(annots_path+x_test[j].split('/')[-2]+'/'+m,newpath2+folder+'_'+n+'.xml')
		print('-------------------- Test split for %s done --------------------'%(lines[i]))
		#For train
		for j in range(len(x_train)):
			m = x_train[j].split('/')[-1].split('.')[0]
			# print(annots_path+Images[j].split('/')[-1].split('.')[0])
			d = x_train[j].split('/')[-2].split('-')
			# print(d)
			filename = x_train[j].split('/')[-1].split('_')[1]
			if len(d)>2:
				folder = '_'.join(d[1:len(d)])
			else:
				folder = d[1]
			n = filename.split('.')[0]
			# print(x_train[j])
			# print(newpath3+folder+'-'+filename)
			# print(annots_path+x_train[j].split('/')[-2]+'/'+m)
			# print(newpath4+folder+'-'+n+'.xml','\n')
			# print(j,'\n')
			os.rename(x_train[j],newpath3+folder+'_'+filename)
			os.rename(annots_path+x_train[j].split('/')[-2]+'/'+m,newpath4+folder+'_'+n+'.xml')
		print('-------------------- Train split for %s done --------------------'%(lines[i]))

def bbox_coord(img_path, annots_path, s):
	path2 = img_path+"*.jpg"
	images = np.array(glob.glob(path2))
	# os.mkdir(os.getcwd()+'/%s_img_resized'%(s))
	for i in range(len(images)):
		filename = images[i].split('/')[-1].split('.')[0]
		img = cv2.imread(images[i])
		resized_img = cv2.resize(img,(416,416),interpolation=cv2.INTER_CUBIC)
		# print(os.getcwd()+'/%s_img_resized/'%(s)+filename+'.jpg')
		cv2.imwrite(os.getcwd()+'/%s_img/'%(s)+filename+'.jpg',resized_img)
		h,w,c = img.shape
		# print(annots_path+filename+'.xml')
		tree = ET.parse(annots_path+filename+'.xml')
		root = tree.getroot()
		# print(filename+str('.jpg'))
		root.find('folder').text = 'Train_img'
		root.find('filename').text = filename+str('.jpg')
		tree.write(annots_path+filename+'.xml')
		for obj in root.iter('object'):
			xmlbox = obj.find('bndbox')
			b = (int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text),
				int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text))
			# print(b)
			###getting relative percentages to original image
			x1p = b[0]/w
			x2p = b[1]/w
			y1p = b[2]/h
			y2p = b[3]/h
	        #### getting new co-ordinates in reshaped space ####
			h_,w_,c_ = resized_img.shape
			x1 = int(x1p*w_)
			x2 = int(x2p*w_)
			y1 = int(y1p*h_)
			y2 = int(y2p*h_)
			size = root.find('size')
			size.find('width').text = str(416)
			size.find('height').text = str(416)
			xmlbox.find('xmin').text = str(x1)
			xmlbox.find('xmax').text = str(x2)
			xmlbox.find('ymin').text = str(y1)
			xmlbox.find('ymax').text = str(y2)
			# print(os.getcwd()+'/%s_annots/'%(s)+filename+'.xml')
			tree.write(annots_path+filename+'.xml')
			# print(b)
			# print('original', x1p, x2p, y1p,y2p)
			# print('resized',x1,x2,y1,y2)
			# cv2.rectangle(img,(b[0],b[2]),(b[1],b[3]),(0,255,0),2)
			# cv2.imshow('original',img)
			# cv2.waitKey(0)
			# cv2.rectangle(resized_img,(x1,y1),(x2,y2),(0,255,0),2)
			# cv2.imshow('resized',resized_img)
			# cv2.waitKey(0)
	# os.rename('/home/vmuser/Desktop/retry/Train_annots',os.getcwd()+'/Train_annots_resized' )
	print('-------------------- Changed annots for %s done --------------------'%(s))

def labels(path,s):
	im2 = path+"/*.jpg"
	path = glob.glob(im2)
	list_file = open('%s_imglist.txt'%(s), 'w')
	for i in range(len(path)):
		#print(path[i])
		list_file.write(path[i]+'\n')
	print('-------------------- Created lists for %s done --------------------'%(s))

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(img_path,annots_path, s):
	#os.mkdir(os.getcwd()+'/Labels')
	im2 = annots_path+"*.xml"
	path = glob.glob(im2)
	print(len(path))
	for i in range(len(path)):
		name = path[i].split('/')[-1].split('.')[0]
		out_file = open(img_path+'%s.txt'%(name), 'w')
		# print(path[i],'\n', name)
		tree=ET.parse(path[i])
		root = tree.getroot()
		size = root.find('size')
		w = int(size.find('width').text)
		h = int(size.find('height').text)
		for obj in root.iter('object'):
			difficult = obj.find('difficult').text
			cls = obj.find('name').text
			if cls not in classes or int(difficult) == 1:
				print(path[i])
			cls_id = classes.index(cls)
			xmlbox = obj.find('bndbox')
			b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
			bb = convert((w,h), b)
			out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
	print('-------------------- Created lists for %s done --------------------'%(s))

def prediction_info(path):
	text_file = open(path, "r")
	lines = text_file.read().split('\n')
	#print(lines)
	for j in range(len(lines)-1):
		#print(lines[j].split(' '))
		Class = lines[j].split(' ')[0]
		Confidence = lines[j].split(' ')[1]
		Left = lines[j].split(' ')[2]
		Top = lines[j].split(' ')[3]
		Right = lines[j].split(' ')[4]
		Bottom = lines[j].split(' ')[-1]
		print(Class, Confidence, Left, Top, Right, Bottom)

####Step1:##### -  get subset list or all list
n = Name()

####Step2:#####- do train test split and move files to respective folders
# text = '/home/ten11/Desktop/send3/labels1.txt'
# img_path = '/home/ten11/Desktop/send3/Images/'
# annots_path = '/home/ten11/Desktop/send3/Annotation/'
# moving(text,img_path,annots_path)

####Step3:##### - resize images and create annotations
# train_img_path = '/home/ten11/Desktop/send3/Train_img/'
# train_annots_path = '/home/ten11/Desktop/send3/Train_annots/'
# bbox_coord(train_img_path, train_annots_path, 'Train')
#
# test_img_path = '/home/ten11/Desktop/send3/Test_img/'
# test_annots_path = '/home/ten11/Desktop/send3/Test_annots/'
# bbox_coord(test_img_path, test_annots_path, 'Test')

####Step4:#### - create lists pf training and testing
# train_img_path = '/home/ten11/Desktop/send3/Train_img/'
# test_img_path = '/home/ten11/Desktop/send3/Test_img/'
# labels(train_img_path,'train')
# labels(test_img_path,'test')

####Step5:#### - lists with class and normalized dimensions
# train_img_path = '/home/ten11/Desktop/send3/Train_img/'
# train_annots_path = '/home/ten11/Desktop/send3/Train_annots/'
# convert_annotation(train_img_path,train_annots_path,'train')

#####Step6: #### - prediction prediction_info
# root3 = '/home/vmuser/Desktop/send3/darknet/me/bounding_boxes.txt'
# prediction_info(root3)


#################Old methods#############################
#Labels(n)
#def Labels(f):
	# Dir = os.getcwd()
	# os.mkdir(Dir+'/Labels')
	# with open(f) as x:
	# 	for line in x:
	# 		line = line.strip()
	# 		os.mkdir(Dir+'/Labels/'+str(line))
	# print('-------------------- Label Files Created --------------------')
#def load_Xmls(root1, root2):
	#images = []
	#originalpath = []
	#folder=[]
	#filename=[]
	# os.mkdir(os.getcwd()+'/sub_annots1')
	# for path, subdirs, files in os.walk(root1):
	# 	#print('hello')
	# 	for name in files:
	# 		direc =os.path.join(path, name)
	# 		f = direc.split('/')[-2].split('-')
	# 		if len(f)>2:
	# 			m = '_'.join(f[1:len(f)])
	# 		else:
	# 			m = f[1]
	# 		#newname = direc.split('/')[-1]+'-'+m
	# 		#folder.append(m)
	# 		s = direc.split('/')[-1].split('_')[1]
	# 		#filename.append(m+'_'+s)
	# 		#print(folder,'     ',filename)
	# 		#print(direc.split('/')[-1]+'-'+m)
	# 		#originalpath.append(root2+newname+'.xml')
	# 		newname = m+'_'+s
	# 		print(root2+newname+'.xml')
	# 		os.rename(direc, root2+newname+'.xml')
	# 		#break
	# 		#folder, filename = Add_ext_To_xmls(root2+n[-1]+'.xml')
	# 		#print(folder+"-"+filename)
	#
	# print('-------------------- Moved annots --------------------')
	#return originalpath
#def Rename_images(path):
    # im2 = path+"/*/*.jpg"
    # path = glob.glob(im2)
    # os.mkdir(os.getcwd()+'/sub_data1')
    # newpath = os.getcwd()+'/sub_data1/'
    # print(newpath)
    # for i in range(len(path)):
    #     print(path[i])
    #     #folder = path[i].split('/')[-2].split('-')[1]
    #     #print(path[i].split('/')[-2].split('-'))
    #     d = path[i].split('/')[-2].split('-')
    #     filename = path[i].split('/')[-1].split('_')[1]
    #     if len(d)>2:
    #     	#print(path[i].split('/')[-2].split('-'))
    #     	#print('_'.join(d[1:len(d)]))
    #     	folder = '_'.join(d[1:len(d)])
    #     else:
    #     	folder = d[1]
    #     	#pass
    #     print(newpath+folder+'_'+filename)
	#
    #     #print(newpath+folder+'_'+filename)
    #     os.rename(path[i], newpath+folder+'_'+filename)
#def Train_Test_split(Annotspath, Imagespath):
    ##Need to fix train test split as more things are going into testing
    # path1 = Annotspath+"/*.xml"
    # path2 = Imagespath+"/*.jpg"
    # Annots = np.array(glob.glob(path1))
    # Images = np.array(glob.glob(path2))
	#
    # # split data
    # tr1 = random.sample(range(len(Annots)), int(len(Annots)*.6))
    # te1 = random.sample(range(len(Annots)), len(Annots)-int(len(Annots)*.6))
    # Annots_train = Annots[tr1]
    # Images_train = []
    # InTrain = []
    # Images_test =[]
    # test_data = []
    # print(len(Annots_train), len(Images))
    # for i in range(len(Annots_train)):
    # 	#print(Annots_train[i])
    #     InTrain.append(Annots_train[i].split('/')[-1].split('.')[0])
    # #print(InTrain)
    # for j in range(len(Images)):
    # 	s = Images[j].split('/')[-1].split('.')[0]
    # 	v = []
    # 	#[print(i) for i, s in enumerate(InTrain) if s in InTrain]
	#
    # 	#a in b checks for the exact string in the array but is currently not finding all
    # 	if s in InTrain:
    # 		Images_train.append(Images[j])
    # 	else:
    # 		Images_test.append(Images[j])
    # print(len(InTrain),len(Images_train),len(Images_test))
    #train_set = Annots[tr1]
    #print(train_set)
    #for i in range(len(path1)):
        #print(Annots[i],'\n',Images[i])
#def Add_ext_To_xmls(train_path, test_path):
	# im2 = filepath+"/*.xml"
	# path = glob.glob(train_path+'*.xml')
	# path1 = glob.glob(test_path+'*.xml')
	# #print(path)
	# for i in range(5):
	# 	print(path[i])
	# 	#folder = path[i].split('/')[-1].split('-')[1].split('.')[0]
	# 	#filename = folder+'_'+path[i].split('/')[-1].split('-')[0].split('_')[1]
	# 	filename = path[i].split('/')[-1].split('.')[0]
	# 	print(filename+str('.jpg'))
	# 	#print(path[i].split('/')[-1].split('.')[0])
	# 	data = ET.parse(path[i])
	# 	root = data.getroot()
	# 	print(root.find('folder').text)
		# root.find('folder').text = 'Train_img'
		# root.find('filename').text = filename+str('.jpg')
		# data.write(path[i])
	# print('-------------------- Added extensions in Xmls --------------------')
	#return folder, filename
