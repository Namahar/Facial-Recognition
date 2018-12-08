import cv2
import os
import numpy as np

# read training images
def prepare_training_data(data_folder_path):
	# get directories
	dirs = os.listdir(data_folder_path)
	faces = []
	labels = []
	
	for dir_name in dirs:
		label = int(dir_name)
	
		subject_dir_path = data_folder_path + "/" + dir_name
		
		# get images in folder
		subject_images_names = os.listdir(subject_dir_path)
		
		for image_name in subject_images_names:
			image_path = subject_dir_path + "/" + image_name
		
			# read image
			image = cv2.imread(image_path)
		
			# detect face
			face, rect = detect_face(image)
		
			if face is not None:
				faces.append(face)
				labels.append(label)
			
	return faces, labels

# detect faces
def detect_face(img):

	# convert image to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	# load opencv face detector
	face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	faces = face_cas.detectMultiScale(
	gray, 
	scaleFactor = 1.1, 
	minNeighbors = 5,
	minSize = (30,30)
	)
		
	# if no faces, return image
	if (len(faces) == 0):
		return None, None
		
	# get face
	x, y, w, h = faces[0]
		
	# return face part only
	return gray[y: y+w, x: x+h], faces[0]
	
def draw_rectangle(img, rect):
	x, y, w, h = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
	print(text)	
	
def predict(test_img):
	# make copy of image
	img = test_img.copy()
	
	# get face
	face, rect = detect_face(img)
	
	if face is None:
		return test_img
	
	# get predicted label
	label = face_recognizer.predict(face)
	label_text = subjects[label[0]]
	
	draw_rectangle(img, rect)
	draw_text(img, label_text, rect[0], rect[1]-5)
		
	return img
	
	
	
subjects = ["", "Saeed", "Nisa", "Anwar"]
	
# create LBPH face recognizer
# local binary pattern histograms
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

faces, labels = prepare_training_data("images")
#print("Total faces: ", len(faces))
#print("Total labels: ", len(labels))
	
# train face recognizer
face_recognizer.train(faces, np.array(labels))

# start video feed	
vc = cv2.VideoCapture(0)

for i in range(100):
	ret, frame = vc.read()
	
	predicted_img = predict(frame)
	
	key = cv2.waitKey(100)
	cv2.imshow("preview", predicted_img)

	if key == 27: # exit on ESC
		break

vc.release()
cv2.destroyAllWindows()


