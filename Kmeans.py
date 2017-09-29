import random
import numpy as np

class kMeans:
	def __init__(self, k, inputs, n):
		self.k = k
		self.inputs = inputs
		self.n = n
		self.currCentroids = []
		self.oldCentroids = []

	def initializeCentroids(self):
		''' Produce k vectors of n random inputs '''
		for i in range(self.k):
			randomVector = []
			for j in range(self.n):
				randomVector.append(random.uniform(-10, 10)) #This range should be same as one in trainingArray.py
			self.currCentroids.append(randomVector)
			self.oldCentroids.append([])

	def assignInputs(self):
		''' For each input vector, assign that vector to the closest mu '''
		self.oldCentroids = self.currCentroids
		clusterVectors = []
		for j in range(len(self.currCentroids)):
			clusterVectors.append([])
		for vector in self.inputs:
			minDist = 9999999999
			cluster = -1
			for i in range(len(self.currCentroids)):
				dist = self.calcDistance(vector, self.currCentroids[i])
				if dist < minDist:
					minDist = dist
					cluster = i
			clusterVectors[cluster].append(vector)
		self.currCentroids = self.findMeanVectors(clusterVectors)

	def findMeanVectors(self, inputVector):
		''' Find the mean vectors for each cluster, then return that vector '''
		meanVectors = []
		for cluster in inputVector:
			if cluster == []:
				randVector = []
				for j in range(self.n):
					randVector.append(random.uniform(-10, 10))
				meanVectors.append(randVector)
			else:
				meanVector = np.mean(cluster, axis = 0)
				meanVectors.append(meanVector)
		return meanVectors

	def calcDistance(self, vector1, vector2):
		''' Return the distance of ||vector1 - vector2|| '''
		return np.linalg.norm(np.subtract(vector1, vector2))

	def hasConverged(self):
		''' If current centroids = old centroids, we have convergence '''
		for i in range(len(self.currCentroids)):
			if np.all(self.currCentroids[i] != self.oldCentroids[i]):
				return False
		return True

	def calculateKMeans(self):
		''' Call the helper functions until we have convergence, then
			return the vector of centroids '''
		self.initializeCentroids()
		while not self.hasConverged():
			self.assignInputs()
		return self.currCentroids

	

if __name__ == '__main__':
	
	testInput = []
	for i in range(1000):
		sample = []
		for j in range(3):
			sample.append(random.uniform(-10, 10))
		testInput.append(sample)
	thing = kMeans(15, testInput, 3)
	#thing.initializeCentroids()
	#for item in thing.currCentroids:
	#	print (item)
	#print (" ")
	
	testK = thing.calculateKMeans()
	for test in testK:
		print (test)