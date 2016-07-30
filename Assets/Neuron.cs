using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class Neuron {

	// Numero de inputs de la Neurona.
	public int i_NumInputs;
	// Un peso para cada input.
	public float[] f_ArrWeights;
	// El numero de activacion de la neurona.
	public float f_Activation;
	// Error Value.
	public float f_ErrorValue;

	public Neuron (int p_NumInputs) {

		// Un peso adicional para el Bias.
		f_Activation = 0;
		i_NumInputs = p_NumInputs + 1;
		f_ArrWeights = new float[i_NumInputs];

		for (int i = 0; i < i_NumInputs; i++) {
		
			// Peso con un valor inicial entre -1 y 1.
			f_ArrWeights[i] = Random.Range(-1f, 1f);
		}
	}
}
