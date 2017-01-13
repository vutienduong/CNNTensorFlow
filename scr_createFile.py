from os import listdir
from os.path import isfile, join

WRITE_FILE = False

if WRITE_FILE:
	print "Write file mris.txt"
	mypath = "/media/sf_LEARN/4th SEMESTER/TensorFlow/JH41/resizeMRI";
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	files = sorted(onlyfiles)
	thefile = open('/media/sf_LEARN/4th SEMESTER/TensorFlow/JH41/mris.txt', 'w')
	for item in files:
	  thefile.write("%s\n" % item)

	thefile.close()


READ_LABEL = True
if READ_LABEL:
	print "Read file mris_label.txt"
	label_file ='/media/sf_LEARN/4th SEMESTER/TensorFlow/JH41/mris_label.txt'
	f = open(label_file, 'r')
	# with open(label_file, 'r') as f:
	#	read_data = f.read()
	#f.closed
	labels = []
	for line in f:
		labels.append(int(line))

	f.closed
	print (labels)
	print (type(labels).__name__)



#from os import walk
#f = []
#for (dirpath, dirnames, filenames) in walk(mypath):
#    f.extend(filenames)
#    break