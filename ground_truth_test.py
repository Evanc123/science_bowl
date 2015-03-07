kaggle_test = np.load("/home/ecater/Desktop/caffe/examples/science_bowl/data/X_kaggle_test.npy")
print kaggle_test.shape
out = net.forward_all(kaggle_test)
np.save("kaggle_test_prob.csv", out['prob'])


"""f = h5py.File('/home/ecater/Desktop/caffe/examples/science_bowl/data/test.h5', 'r')
X = np.asarray(f['data'])
y = np.asarray(f['label'])
print y, X.shape
#out = net.forward_all(data=np.asarray(X))
out = np.load('test_pred.npy')
number_right = 0
for i in range(out.shape[0]):
    max_num = 0
    highest_index = 0
    for j in range(out.shape[1]):
        if out[i][j][0][0] > max_num:
            highest_index = j
	    max_num = out[i][j][0][0]
    if y[i] == highest_index:
        number_right += 1

print number_right / out.shape[0]
