import time
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]
import numpy as np
from evaluation import compute_rsme, compute_nlpd, compute_CoBEAU, compute_coverage_probability
num_meta_epochs = 3


def visualise(net, datasetcreator, num_meta_epochs=num_meta_epochs, plot=True):

    dataset = datasetcreator()
    X_train, y_train = dataset.train_dataset

    X_test, y_test = dataset.test_dataset

    test_idx = dataset.return_test_idx
    train_idx = dataset.return_train_idx

    start_time = time.time()

    y_pred = net.predict(X_test)
    #try: y_pred[1]
    #except:
    #        y = y_pred[0]

    #if plot:
    fig = plt.figure(figsize=(20, 5 * num_meta_epochs + 1))
    fig.add_subplot(num_meta_epochs + 1, 1, 1)
    plt.plot(X_test[test_idx], y_pred[test_idx], c='orange',
             label='initial prediction')
    plt.scatter(X_test, y_test)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')

    pred_list = [y_pred]
    error_list = [np.mean((y_pred - y_test)**2)]

    for i in range(num_meta_epochs):

        #after training
        net.fit(X_train, y_train)  #,shuffle=True)
        y_pred = net.predict(X_test)
        #if len(np.array(y_pred).shape)>1:
        #    y_pred = y_pred[0]
        #try: y_pred[1]
        #except:
        #    y = y_pred[0]

        pred_list.append(y_pred)
        if plot:
            fig.add_subplot(num_meta_epochs + 1, 1, i + 2)
            plt.plot(X_test[test_idx], y_pred[test_idx], c='orange')
            plt.scatter(X_test, y_test)
            #fig.show()
            plt.xlabel('x')
            plt.ylabel('y')

    fig2 = plt.figure(figsize=(20, 5))

    for i, pred in enumerate(pred_list):
        error_list.append(compute_rsme(pred, y_test))
        plt.scatter(X_test, y_test)
        plt.plot(X_test[test_idx], pred[test_idx],
                 label='prediction after {} epochs'.format(i))
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
    fig3 = plt.figure(figsize=(20, 5))
    plt.plot(error_list)
    plt.xlabel('x')
    plt.ylabel('y')
    return 'this run of {} epochs and {} metaepochs took {}s'.format(
        net.num_epochs, num_meta_epochs, time.time() - start_time)


#num_epochs = 100
#num_meta_epochs = 5 #also numplots
#learning_rate = 1/num_epochs


def visualise_uncertainty(net, datasetcreator, num_meta_epochs=num_meta_epochs,
                          plot=True):
    start_time = time.time()

    dataset = datasetcreator()
    X_train, y_train = dataset.train_dataset

    X_test, y_test = dataset.test_dataset

    test_idx = dataset.return_test_idx
    train_idx = dataset.return_train_idx

    y_pred, y_var = net.get_mean_and_std(X_test)

    pred_list = [y_pred]
    var_list = [y_var]
    error_list = [np.mean((y_pred - y_test)**2)]
    nlpd_list = [compute_nlpd(y_pred, y_test, y_var)]
    cobeau_list = [compute_CoBEAU(y_pred, y_test, y_var)]
    covprob_list = [compute_coverage_probability(y_pred, y_test, y_var)]

    fig = plt.figure(figsize=(20, 5 * num_meta_epochs + 1))

    if plot:
        fig.add_subplot(num_meta_epochs + 1, 1, 1)

        pred_mean, pred_std = net.get_mean_and_std(X_test)

        plt.plot(X_test[test_idx], pred_mean[test_idx], c='orange')
        plt.fill_between(
            X_test[test_idx].ravel(), pred_mean[test_idx].ravel(),
            pred_mean[test_idx].ravel() + pred_std[test_idx].ravel(), alpha=.3,
            color='b')
        plt.fill_between(
            X_test[test_idx].ravel(), pred_mean[test_idx].ravel(),
            pred_mean[test_idx].ravel() - pred_std[test_idx].ravel(), alpha=.3,
            color='b')
        plt.scatter(X_test, y_test)
        plt.title('original prediction')
        plt.xlabel('x')
        plt.ylabel('y')

    for i in range(num_meta_epochs - 1):

        #after training
        net.fit(X_train, y_train)  #,shuffle=True)
        pred_mean, pred_std = net.get_mean_and_std(X_test)

        pred_list.append(pred_mean)
        var_list.append(pred_std)

        if plot:
            fig.add_subplot(num_meta_epochs + 1, 1, i + 2)
            plt.plot(X_test[test_idx], pred_mean[test_idx], c='orange')
            plt.fill_between(
                X_test[test_idx].ravel(), pred_mean[test_idx].ravel(),
                pred_mean[test_idx].ravel() + pred_std[test_idx].ravel(),
                alpha=.3, color='b')
            plt.fill_between(
                X_test[test_idx].ravel(), pred_mean[test_idx].ravel(),
                pred_mean[test_idx].ravel() - pred_std[test_idx].ravel(),
                alpha=.3, color='b')
            plt.scatter(X_test, y_test)
            plt.title('prediction after {} meta-epochs with {} epochs'.format(
                i, net.num_epochs))
            plt.xlabel('x')
            plt.ylabel('y')

    fig.add_subplot(num_meta_epochs + 1, 1, num_meta_epochs + 1)
    plt.plot(X_test[test_idx], pred_mean[test_idx], c='orange')
    plt.fill_between(X_test[test_idx].ravel(), pred_mean[test_idx].ravel(),
                     pred_mean[test_idx].ravel() + pred_std[test_idx].ravel(),
                     alpha=.3, color='b')
    plt.fill_between(X_test[test_idx].ravel(), pred_mean[test_idx].ravel(),
                     pred_mean[test_idx].ravel() - pred_std[test_idx].ravel(),
                     alpha=.3, color='b')
    plt.scatter(X_test, y_test)
    plt.xlabel('x')
    plt.ylabel('y')
    #fig2 = plt.figure(figsize=(20,5))
    i = 0
    for pred, std in zip(pred_list, var_list):
        i += 1
        error_list.append(compute_rsme(pred, y_test))
        nlpd_list.append(compute_nlpd(pred, y_test, std))
        cobeau_list.append(compute_CoBEAU(pred, y_test, std))
        covprob_list.append(compute_coverage_probability(pred, y_test, std))

        plt.scatter(X_test, y_test)
        plt.plot(X_test[test_idx], pred[test_idx],
                 label='prediction after {} epochs'.format(i))
        plt.legend()
        plt.title('Development of predictions through time')
        plt.xlabel('x')
        plt.ylabel('y')

    fig3 = plt.figure(figsize=(20, 5))
    plt.plot(error_list)
    plt.title('error through time')
    plt.xlabel('epoch')
    plt.ylabel('error')

    fig4 = plt.figure(figsize=(20, 5))
    plt.plot(nlpd_list)
    plt.title('nlpd through time')
    plt.xlabel('epoch')
    plt.ylabel('nlpd')

    fig5 = plt.figure(figsize=(20, 5))
    print(cobeau_list)
    #print(cobeau_list[:, 1])
    p_list = [entry[1] for entry in cobeau_list]
    r_list = [entry[0] for entry in cobeau_list]
    plt.plot(r_list, label='r')
    plt.plot(p_list, label='p')
    plt.title('CoBEAU through time')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('nlpd')

    fig6 = plt.figure(figsize=(20, 5))
    plt.plot(covprob_list)
    plt.title('Coverage through time')
    plt.xlabel('epoch')
    plt.ylabel('nlpd')
    return 'this run of {} epochs and {} metaepochs took {}s'.format(
        net.num_epochs, num_meta_epochs, time.time() - start_time)
