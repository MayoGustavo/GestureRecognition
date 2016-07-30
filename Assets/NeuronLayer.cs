using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class NeuronLayer {

	// Numero de Neuronas en la capa.
	public int i_NumNeurons;
	// Array que almacena las neuronas de la capa.
	public List<Neuron> l_Neurons;


	public NeuronLayer (int p_NumNeurons, int p_NumInputsXNeuron) {

		i_NumNeurons = p_NumNeurons;
		l_Neurons = new List<Neuron> ();

		for (int i = 0; i < p_NumNeurons; i++) {

			l_Neurons.Add(new Neuron(p_NumInputsXNeuron));
		}
	}
}
