train-nn: neuralnet/NeuralNet.py neuralnet/Dataset.py
	python3 neuralnet/NeuralNet.py

train-nn2: neuralnet/NeuralNet2.py neuralnet/Dataset.py
	python3 neuralnet/NeuralNet2.py

train-nn3: neuralnet/NeuralNet3.py neuralnet/Dataset.py
	python3 neuralnet/NeuralNet3.py

train-nn4: neuralnet/NeuralNet4.py neuralnet/Dataset.py
	python3 neuralnet/NeuralNet4.py

clean:
	rm -r model/* performance/* weights/*