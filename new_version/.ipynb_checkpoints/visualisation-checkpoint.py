import time
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]
import numpy as np

num_meta_epochs = 3

def visualise(net,datasetcreator,num_meta_epochs=num_meta_epochs,plot=True):

	dataset = datasetcreator()
	X_train, y_train = dataset.train_dataset

	X_test, y_test = dataset.test_dataset

	test_idx=dataset.return_test_idx
	train_idx=dataset.return_train_idx


	start_time = time.time()

	y_pred = net.predict(X_test)
	#try: y_pred[1]
	#except:
	#        y = y_pred[0]

	#if plot:
	fig = plt.figure(figsize=(20,5*num_meta_epochs+1))
	fig.add_subplot(num_meta_epochs+1,1,1)
	plt.plot(X_test[test_idx],y_pred[test_idx],c='orange',label='initial prediction')
	plt.scatter(X_test,y_test)
	plt.legend()



	pred_list = [y_pred]
	error_list = [np.mean((y_pred - y_test)**2)]


	for i in range(num_meta_epochs):

		#after training
		net.fit(X_train,y_train)#,shuffle=True)
		y_pred = net.predict(X_test)
		#if len(np.array(y_pred).shape)>1:
		#    y_pred = y_pred[0]
		#try: y_pred[1]
		#except:
		#    y = y_pred[0]

		pred_list.append(y_pred)
		if plot:
			fig.add_subplot(num_meta_epochs+1,1,i+2)
			plt.plot(X_test[test_idx],y_pred[test_idx],c='orange')
			plt.scatter(X_test,y_test)
			#fig.show()

	fig2 = plt.figure(figsize=(20,5))

	for i,pred in enumerate(pred_list):
		error_list.append(np.mean((pred-y_test)**2))
		plt.scatter(X_test,y_test)
		plt.plot(X_test[test_idx],pred[test_idx],label='prediction after {} epochs'.format(i))
		plt.legend()
	fig3 = plt.figure(figsize=(20,5))
	plt.plot(error_list)
	return 'this run of {} epochs and {} metaepochs took {}s'.format(net.num_epochs, num_meta_epochs,time.time() - start_time)


#num_epochs = 100
#num_meta_epochs = 5 #also numplots
#learning_rate = 1/num_epochs


def visualise_uncertainty(net,datasetcreator,num_meta_epochs=num_meta_epochs,plot=True):
	start_time = time.time()


	dataset = datasetcreator()
	X_train,y_train = dataset.train_dataset

	X_test,y_test = dataset.test_dataset

	test_idx=dataset.return_test_idx
	train_idx=dataset.return_train_idx


	y_pred,_ = net.get_mean_and_std(X_test)

	pred_list = [y_pred]
	error_list = [np.mean((y_pred - y_test)**2)]

	fig = plt.figure(figsize=(20,5*num_meta_epochs+1))

	if plot:
		fig.add_subplot(num_meta_epochs+1,1,1)

		pred_mean,pred_std = net.get_mean_and_std(X_test)

		plt.plot(X_test[test_idx],pred_mean[test_idx],c='orange')
		plt.fill_between(X_test[test_idx].ravel(), pred_mean[test_idx].ravel(),
						 pred_mean[test_idx].ravel()+pred_std[test_idx].ravel(), alpha=.3, color='b')
		plt.fill_between(X_test[test_idx].ravel(), pred_mean[test_idx].ravel(),
						 pred_mean[test_idx].ravel()-pred_std[test_idx].ravel(), alpha=.3, color='b')
		plt.scatter(X_test,y_test)
	for i in range(num_meta_epochs-1):

		#after training
		net.fit(X_train,y_train)#,shuffle=True)
		pred_mean,pred_std = net.get_mean_and_std(X_test)

		pred_list.append(pred_mean)
		if plot:
			fig.add_subplot(num_meta_epochs+1,1,i+2)
			plt.plot(X_test[test_idx],pred_mean[test_idx],c='orange')
			plt.fill_between(X_test[test_idx].ravel(), pred_mean[test_idx].ravel(),
							 pred_mean[test_idx].ravel()+pred_std[test_idx].ravel(), alpha=.3, color='b')
			plt.fill_between(X_test[test_idx].ravel(), pred_mean[test_idx].ravel(),
							 pred_mean[test_idx].ravel()-pred_std[test_idx].ravel(), alpha=.3, color='b')
			plt.scatter(X_test,y_test)

	fig.add_subplot(num_meta_epochs+1,1,num_meta_epochs+1)
	plt.plot(X_test[test_idx],pred_mean[test_idx],c='orange')
	plt.fill_between(X_test[test_idx].ravel(), pred_mean[test_idx].ravel(),
					 pred_mean[test_idx].ravel()+pred_std[test_idx].ravel(), alpha=.3, color='b')
	plt.fill_between(X_test[test_idx].ravel(), pred_mean[test_idx].ravel(),
					 pred_mean[test_idx].ravel()-pred_std[test_idx].ravel(), alpha=.3, color='b')
	plt.scatter(X_test,y_test)
	#fig2 = plt.figure(figsize=(20,5))

	#for i,pred in enumerate(pred_list):
	#    error_list.append(np.mean((pred-y_test)**2))
	#    plt.scatter(X_test,y_test)
	#    plt.plot(X_test[test_idx],pred[test_idx],label='prediction after {} epochs'.format(i))
	#    plt.legend()
	#fig3 = plt.figure(figsize=(20,5))
	#plt.plot(error_list)
	return 'this run of {} epochs and {} metaepochs took {}s'.format(net.num_epochs, num_meta_epochs,time.time() - start_time)
