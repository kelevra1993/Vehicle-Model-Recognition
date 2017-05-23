#Author: Jacob Gildenblat, 2014
#License: you may use this for whatever you like 
import logging
import sys, glob, argparse
import numpy as np
import math, cv2
from scipy.stats import multivariate_normal
import time
from sklearn import svm
from sklearn import mixture

use_sklearn = True
logger = logging.getLogger(__name__)

def dictionary(descriptors, N):    
    em = cv2.EM(N)
    print descriptors.shape
    em.train(descriptors)

    return np.float32(em.getMat("means")), \
        np.float32(em.getMatVector("covs")), np.float32(em.getMat("weights"))[0]

def image_descriptors(file):
    img = cv2.imread(file, 0)
    img = cv2.resize(img, (256, 256))
    _ , descriptors = cv2.SIFT().detectAndCompute(img, None)
    return descriptors

def folder_descriptors(folder):
    files = glob.glob(folder + "/*.jpg")
    print("Calculating descriptos. Number of images is", len(files))
    return np.concatenate([image_descriptors(file) for file in files])

def likelihood_momentOld(x, ytk, moment):    
    # print "likelihood_momentOld01 :", x.shape
    x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
    # print "likelihood_momentOld01 : ", x_moment, ytk
    return x_moment * ytk

def likelihood_moment(x, ytk, moment):    

    x_moment = np.power(np.float32(x), moment) if moment > 0 else np.ones(x.shape[0]).reshape(x.shape[0], 1)
    # print "likelihood_moment", x_moment.shape, ytk.shape
    return x_moment * ytk.reshape(ytk.shape[0], 1)
    
    
def likelihood_statisticsOld(samples, means, covs, weights):
    gaussians, s0, s1,s2 = {}, {}, {}, {}
    rsamples = zip(range(0, len(samples)), samples)
    
    g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights)) ]    
        
    for index, x in rsamples:        
        gaussians[index] = np.array([g_k.pdf(x) for g_k in g])
        
    for k in range(0, len(weights)):
        s0[k], s1[k], s2[k] = 0, 0, 0
        for index, x in rsamples:
            probabilities = np.multiply(gaussians[index], weights)
            probabilities = probabilities / np.sum(probabilities)
            s0[k] = s0[k] + likelihood_momentOld(x, probabilities[k], 0)
            s1[k] = s1[k] + likelihood_momentOld(x, probabilities[k], 1)
            s2[k] = s2[k] + likelihood_momentOld(x, probabilities[k], 2)

    return s0, s1, s2
    
gg = None
def likelihood_statistics(samples, means, covs, weights):
    # gaussians, s0, s1,s2 = {}, {}, {}, {}
    # rsamples = zip(range(0, len(samples)), samples)
    global gg
    if gg is None:
        gg = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights)) ]
    g = gg
    
    # print "Samples", samples.shape, g[0].pdf(samples).shape
    # for index, x in rsamples:        
        # gaussians[index] = np.array([g_k.pdf(x) for g_k in g])
    
    tt =  g[0].pdf(samples)     
    for g_k in g[1:]:                      
        p = g_k.pdf(samples)        
        tt = np.vstack((tt, p))
    # print "XX", tt.shape, tt[:,0].shape, gaussians[0].shape
    
    rw = np.reshape(weights, (weights.shape[0], 1))
    
    ss0 = []
    ss1 = []
    ss2 = []
    for k in range(0, len(weights)):
        # s0[k], s1[k], s2[k] = 0, 0, 0
        
        probabilities = np.multiply(tt, rw)                        
        probabilities = probabilities / np.sum(probabilities, axis=0)
        lm = likelihood_moment(samples, probabilities[k], 0)
        ss0.append(np.sum(lm, axis=0))
        lm = likelihood_moment(samples, probabilities[k], 1)
        ss1.append(np.sum(lm, axis=0))
        lm = likelihood_moment(samples, probabilities[k], 2)
        ss2.append(np.sum(lm, axis=0))
        
        # for index, x in rsamples:
            # probabilitiesOld = np.multiply(gaussians[index], weights)
            # probabilitiesOld = probabilitiesOld / np.sum(probabilitiesOld)            
            # s0[k] = s0[k] + likelihood_momentOld(x, probabilitiesOld[k], 0)
            # s1[k] = s1[k] + likelihood_momentOld(x, probabilitiesOld[k], 1)
            # s2[k] = s2[k] + likelihood_momentOld(x, probabilitiesOld[k], 2)                        
       
    ss0 = np.asarray(ss0)
    ss1 = np.asarray(ss1)
    ss2 = np.asarray(ss2)
    # prin nt
    # print "ss0", ss0[3], s0[3]
    # print "ss1", ss1[3], s1[3]
    # print "ss2", ss2[3], s2[3]
    # exit(1)
    return np.reshape(ss0, (ss0.shape[0], 1)), ss1, ss2

def fisher_vector_weightsOld(s0, s1, s2, means, covs, w, T):
    return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, len(w))])

def fisher_vector_weights(s0, s1, s2, means, covs, w, T):       
    return (s0 - T * w) / np.sqrt(w)        

def fisher_vector_meansOld(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])    

def fisher_vector_means(s0, s1, s2, means, sigma, w, T):    
    return (s1 - means * s0) / (np.sqrt(np.multiply(sigma, w)))
    

def fisher_vector_sigmaOld(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])
    
def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
    return (s2 - 2 * means * s1 + (means * means - sigma) * s0) / (np.sqrt(2*w)*sigma)         

def normalize(fisher_vector):
    v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
    return v / np.sqrt(np.dot(v, v))

def fisher_vectorOld(samples, means, covs, w):
    s0, s1, s2 =  likelihood_statisticsOld(samples, means, covs, w)    
    # print "s0,s1,s2", a.shape, b.shape, c.shape
    # return None
    T = samples.shape[0]
    covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
    a = fisher_vector_weightsOld(s0, s1, s2, means, covs, w, T)
    b = fisher_vector_meansOld(s0, s1, s2, means, covs, w, T)
    c = fisher_vector_sigmaOld(s0, s1, s2, means, covs, w, T)
    # print "w,m,s", a.shape, b.shape, c.shape
    fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
    # print a.shape, fv.shape
    fv = normalize(fv)
    # exit(1)
    return fv

def fisher_vectorNew(samples, means, covs, w):    
    s0, s1, s2 =  likelihood_statistics(samples, means, covs, w)    
    # return None
    T = samples.shape[0]
    # print covs[0]
    covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
    # print covs.shape
    # s0 =  np.float32(map(lambda x:x[0], s0.values()))
    s0 = np.reshape(s0, (s0.shape[0], 1))    
    # s1 = np.float32(map(lambda x:x, s1.values()))
    # s2 = np.float32(map(lambda x:x, s2.values()))
    w = np.reshape(w, (w.shape[0], 1))
    
    
    a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
    # bOld = fisher_vector_meansOld(s0, s1, s2, means, covs, w, T)
    b = fisher_vector_means(s0, s1, s2, means, covs, w, T)        
    # cOld = fisher_vector_sigmaOld(s0, s1, s2, means, covs, w, T)
    c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)    
    
    # print "x", a.shape, b.shape, c.shape
    fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
    # print fv.shape    
    # print a.shape, fv.shape
    fv = normalize(fv)    
    return fv    

def fisher_vector(samples, means, covs, w):
    # fvOld = fisher_vectorOld(samples, means, covs, w)
    fv =  fisher_vectorNew(samples, means, covs, w)
    # print fvOld.shape, fv.shape
    # d = (fvOld-fv)
    # print "Diff", d.dot(d)
    return fv
    
def generate_gmm(N, input_folder = None, descs=None):
    if input_folder is not None:
        words = np.concatenate([folder_descriptors(folder) for folder in glob.glob(input_folder + '/*')]) 
    elif descs is not None:
        words = descs
     
    
    logger.info("Training GMM of size %d with %s descriptors of size %d", N, words.shape[0], words.shape[1])
    if use_sklearn:    
        gmm = mixture.GaussianMixture(n_components=N, verbose=1)
        gmm.fit(words)
        return gmm.means_, gmm.covariances_, gmm.weights_
    else:
        means, covs, weights = dictionary(words, N)
        #Throw away gaussians with weights that are too small:
        th = 1.0 / N
        means = np.float32([m for k,m in zip(range(0, len(weights)), means) if weights[k] > th])
        covs = np.float32([m for k,m in zip(range(0, len(weights)), covs) if weights[k] > th])
        weights = np.float32([m for k,m in zip(range(0, len(weights)), weights) if weights[k] > th])

        # np.save("means.gmm", means)
        # np.save("covs.gmm", covs)
        # np.save("weights.gmm", weights)
        return means, covs, weights

def get_fisher_vectors_from_folder(folder, gmm):
    files = glob.glob(folder + "/*.jpg")
    return np.float32([fisher_vector(image_descriptors(file), *gmm) for file in files])

def fisher_features(folder, gmm):
    folders = glob.glob(folder + "/*")
    features = {f : get_fisher_vectors_from_folder(f, gmm) for f in folders}
    return features

def train(gmm, features):
    X = np.concatenate(features.values())
    Y = np.concatenate([np.float32([i]*len(v)) for i,v in zip(range(0, len(features)), features.values())])

    clf = svm.SVC()
    clf.fit(X, Y)
    return clf

def success_rate(classifier, features):
    print("Applying the classifier...")
    X = np.concatenate(np.array(features.values()))
    Y = np.concatenate([np.float32([i]*len(v)) for i,v in zip(range(0, len(features)), features.values())])
    res = float(sum([a==b for a,b in zip(classifier.predict(X), Y)])) / len(Y)
    return res
    
def load_gmm(folder = ""):
    files = ["means.gmm.npy", "covs.gmm.npy", "weights.gmm.npy"]
    return map(lambda file: load(file), map(lambda s : folder + "/" , files))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d' , "--dir", help="Directory with images" , default='.')
    parser.add_argument("-g" , "--loadgmm" , help="Load Gmm dictionary", action = 'store_true', default = False)
    parser.add_argument('-n' , "--number", help="Number of words in dictionary" , default=5, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':    
    args = get_args()
    working_folder = args.dir

    gmm = load_gmm(working_folder) if args.loadgmm else generate_gmm(args.number, input_folder=working_folder)
    fisher_features = fisher_features(working_folder, gmm)
    #TBD, split the features into training and validation
    classifier = train(gmm, fisher_features)
    rate = success_rate(classifier, fisher_features)
    print("Success rate is", rate)