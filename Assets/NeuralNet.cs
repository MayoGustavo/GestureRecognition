using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;

public class NeuralNet : MonoBehaviour {

	public int i_NumInputs;
	public int i_NumOutputs;
	public int i_NumHiddenLayers;
	public int i_NumNeuronsXHiddenLayer;

	private List<NeuronLayer> nl_Layers = new List<NeuronLayer>();

	public int BIAS = -1;
	public float RESPONSE = 0.5f;
	public float LEARNING_RATE = 0.1f;
	public float ERROR_THRESHOLD = 0.003f;

	float f_ErrorSum;
	bool b_Trained;
	int i_EpochCount;

	void Start () {

		f_ErrorSum = 1f;
		b_Trained = false;
		i_EpochCount = 0;
	}

	public bool Trained () {

		return b_Trained;
	}

	public float Error () {

		return f_ErrorSum;
	}

	public int Epoch () {

		return i_EpochCount;
	}

	// Crea las layers con su estructura de neuronas.
	public void CreateNet ()
	{
		nl_Layers = new List<NeuronLayer>();

		if (i_NumHiddenLayers > 0) {
		
			// Primer capa oculta.
			nl_Layers.Add(new NeuronLayer(i_NumNeuronsXHiddenLayer,i_NumInputs));
		
			// Otras capas ocultas, en caso de haberlas.
			for (int i = 0; i < i_NumHiddenLayers - 1; i++) {

				nl_Layers.Add(new NeuronLayer(i_NumNeuronsXHiddenLayer, i_NumNeuronsXHiddenLayer));
			}

			// Capa Salida.
			nl_Layers.Add(new NeuronLayer(i_NumOutputs, i_NumNeuronsXHiddenLayer));
		}
		else {

			// Capa Salida.
			nl_Layers.Add(new NeuronLayer(i_NumOutputs, i_NumInputs));
		}
	}


	// Calcula los outputs a partir de los inputs. Feedforward.
	public float[] UpdateNeuralNet (float[] f_Inputs)
	{
		List<float> f_Outputs = new List<float>();
		int cWeight = 0;

		if (f_Inputs.Length != i_NumInputs)
			return f_Outputs.ToArray();

		// Por cada Layer.
		for (int LayerCount = 0; LayerCount < i_NumHiddenLayers + 1; LayerCount++) {
		
			if (LayerCount > 0) {

				f_Inputs = f_Outputs.ToArray();
			}

			f_Outputs.Clear();
			cWeight = 0;

			// Por cada Neurona.
			for (int NeuronCount = 0; NeuronCount < nl_Layers[LayerCount].i_NumNeurons; NeuronCount++) {

				float f_NetInput = 0;

				int i_Inputs = nl_Layers[LayerCount].l_Neurons[NeuronCount].i_NumInputs;

				//Por cada peso.
				for (int WeightCount = 0; WeightCount < i_Inputs - 1; WeightCount++) {

					f_NetInput += nl_Layers[LayerCount].l_Neurons[NeuronCount].f_ArrWeights[WeightCount] * f_Inputs[cWeight++];
				}

				// Agregar el Bias.
				f_NetInput += nl_Layers[LayerCount].l_Neurons[NeuronCount].f_ArrWeights[i_Inputs - 1] * BIAS;

				nl_Layers[LayerCount].l_Neurons[NeuronCount].f_Activation = Sigmoid(f_NetInput, RESPONSE);
				f_Outputs.Add(Sigmoid(f_NetInput, RESPONSE));
				cWeight = 0;
			}

		}

		return f_Outputs.ToArray ();
	}

	/// <summary>
	/// Train the ANN.
	/// </summary>
	/// <param name="l_TrainingSet"> Training Set .</param>
	public bool Train (List<float[]> l_TrainingSetIn, List<float[]> l_TrainingSetOut) {

		f_ErrorSum = 1;

		if ((l_TrainingSetIn.Count != l_TrainingSetOut.Count) ||
		    (l_TrainingSetIn[0].Length != i_NumInputs) ||
		    (l_TrainingSetOut[0].Length != i_NumOutputs)) 
		{
			print ("ERROR Inputs != Outputs ERROR");
			return false;
		}

		// Entrenar hasta que ErrorSum sea menor que el threshold.
		while (f_ErrorSum > ERROR_THRESHOLD) {

			if (!NetworkTrainingEpoch (l_TrainingSetIn, l_TrainingSetOut))
				return false;

			//print ("Error Sum: " + f_ErrorSum);
		}

		b_Trained = true;

		return true;
	}

	/// <summary>
	/// Una iteracion de Backpropagation.
	/// </summary>
	/// <returns><c>true</c><c>false</c> Si hubo un problema.</returns>
	/// <param name="SetIn">Inputs.</param>
	/// <param name="SetOut">Expected Outputs.</param>
	bool NetworkTrainingEpoch (List<float[]> SetIn, List<float[]> SetOut) {

		// Valor de error acumulado para el training set.
		f_ErrorSum = 0;

		// Correr cada patron de entrada por la red. Calcular el error y actualizar los pesos.
		for (int vec = 0; vec < SetIn.Count; vec++) {

			// Por cada patro de entrada calcular los pesos.
			float[] outputs = UpdateNeuralNet(SetIn[vec]);

			// Retornar false si hubo error.
			if (outputs.Length == 0)
				return false;

			// Por cada neurona de salida calcular error y corregir pesos.
			for (int op = 0; op < i_NumOutputs; op++) {

				// Calcular el error.
				float f_Error = (SetOut[vec][op] - outputs[op]) * outputs[op] * (1 - outputs[op]);

				// Actualizar el error total. Cuando este valor es menor que el threshold el entrenamiento termina.
				f_ErrorSum += (SetOut[vec][op] - outputs[op]) * (SetOut[vec][op] - outputs[op]);

				// Una sola hidden layer.
				nl_Layers[1].l_Neurons[op].f_ErrorValue = f_Error;

				int biasc = 0;
				// Por cada peso, menos el bias.
				for (int w = 0; w < nl_Layers[1].l_Neurons[op].f_ArrWeights.Length - 1; w++) {

					// Calcular el nuevo peso.
					nl_Layers[1].l_Neurons[op].f_ArrWeights[w] += f_Error * LEARNING_RATE * nl_Layers[0].l_Neurons[w].f_Activation;
					biasc = w;
				}

				// Nuevo peso para el bias.
				biasc++;
				nl_Layers[1].l_Neurons[op].f_ArrWeights[biasc] += f_Error * LEARNING_RATE * BIAS;
			}

			int n = 0;

			// Por cada neurona en la capa oculta calcular el error y reajustar pesos.
			for (int curNbrHid = 0; curNbrHid < nl_Layers[0].l_Neurons.Count; curNbrHid++) {

				float f_Error = 0;

				// Iterar cada neurona de la capa de salida conectadas con esta neurona de la capa oculta.
				for (int curNbrOut = 0; curNbrOut < nl_Layers[1].l_Neurons.Count; curNbrOut++) {

					f_Error += nl_Layers[1].l_Neurons[curNbrOut].f_ErrorValue * nl_Layers[1].l_Neurons[curNbrOut].f_ArrWeights[n];
				}

				// Calcular error.
				f_Error *= nl_Layers[0].l_Neurons[curNbrHid].f_Activation * (1 - nl_Layers[0].l_Neurons[curNbrHid].f_Activation);

				// Por cada peso, calcular el nuevo basado en el error y el learning rate.
				for (int w = 0; w < i_NumInputs; w++) {

					// Calcular nuevo peso.
					nl_Layers[0].l_Neurons[curNbrHid].f_ArrWeights[w] += f_Error * LEARNING_RATE * SetIn[vec][w];
				}

				// Nuevo peso para bias.
				nl_Layers[0].l_Neurons[curNbrHid].f_ArrWeights[i_NumInputs] += f_Error * LEARNING_RATE * BIAS;

				n++;
			}
		}// Siguiente float de inputs.

		return true;
	}


	float Sigmoid (float p_Activation, float p_Response) {

		return 1 / (1 + Mathf.Pow((float)Math.E, -p_Activation / p_Response));
	}
}
