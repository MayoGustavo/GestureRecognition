using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine.UI;

public class Controller : MonoBehaviour {

	public int i_NumSmoothPoints;
	public float MATCH_TOLERANCE = 0.95f;
	public Text t_State;
	public GameObject go_PanelGestureName;
	public Text t_GestureName;

	// Red Neuronal.
	NeuralNet ANN;

	// Listas con los gestos, raw y smooth.
	List<Vector3> l_VecPath;
	List<Vector3> l_VecSmoothPath;
	List<float[]> l_TrainingSetIn;
	List<float[]> l_TargetSetOut;
	List<int> l_CountPerGesture;
	List<string> l_GestureName;
	int i_GestureCount;
	// El trazo suavizado pasado a floats en direcciones.
	float[] f_VecSmoothPath;
	float f_SumOutput;
	int i_BestOut;
	Mode e_Mode;

	enum Mode
	{
		ACTIVE, UNREADY, TRAINING, ERROR, ADDGESTURE, RECOGNIZE
	}

	void Start () {
	
		ANN = GetComponent<NeuralNet> ();
		l_VecPath = new List<Vector3> ();
		l_VecSmoothPath = new List<Vector3> ();
		l_TrainingSetIn = new List<float[]> ();
		l_TargetSetOut = new List<float[]> ();
		l_CountPerGesture = new List<int> ();
		l_GestureName = new List<string> ();
		f_VecSmoothPath = new float[24];
		i_GestureCount = 0;
		e_Mode = Mode.UNREADY;
		t_State.text = "UNREADY";
	}
	
	void Update () {
	
		switch (e_Mode) {

		 case Mode.RECOGNIZE: {

			if (Input.GetMouseButton(0)) {
				AddPoint();
			}
			if (Input.GetMouseButtonUp(0)) {
				if (Smooth()) {
					CreateNormalizedPath();
					CheckGesture();
				}
				ClearMouseData();
			}
			break;
		}
		
		 case Mode.ADDGESTURE: {

			if (Input.GetKeyDown(KeyCode.Insert)) {
				e_Mode = Mode.UNREADY;
				t_State.text = "UNREADY";
				go_PanelGestureName.SetActive(true);
			}

			if (Input.GetMouseButton(0)) {
				AddPoint();
			}
			if (Input.GetMouseButtonUp(0)) {

				if (Smooth()) {
					CreateNormalizedPath();
					SaveNormalizedPath();
				}
				ClearMouseData();
			}

			break;
		}

		}
	}

	void AddPoint () {

		Vector3 v3_WorldPosition = Camera.main.ScreenToWorldPoint (Input.mousePosition);
		v3_WorldPosition.z = 0f;
		l_VecPath.Add (v3_WorldPosition);
	}

	void CheckGesture () {

		// Llamar a la red y guardar outputs, luego comprobar contra el target.
		float[] outputs = ANN.UpdateNeuralNet (f_VecSmoothPath);

		i_BestOut = -1;
		for (int outindex = 0; outindex < outputs.Length; outindex++) {

			if (outputs[outindex] > MATCH_TOLERANCE)
				i_BestOut = outindex;

			print ("OUTPUT: " + outputs[outindex]);
		}

		if (i_BestOut != -1)
			print ("GESTURE NAME: " + l_GestureName [i_BestOut]);
		else 
			print ("UNRECOGNIZED GESTURE");
	}

	void ClearMouseData () {

		l_VecPath.Clear ();
		l_VecSmoothPath.Clear ();
	}

	void CreateNormalizedPath () {

		string s_Cadena = "";
		f_VecSmoothPath = new float[24];
		for (int point = 0; point < l_VecSmoothPath.Count - 1; point++) {

			Vector3 v3_Direction = l_VecSmoothPath[point + 1] - l_VecSmoothPath[point];
			v3_Direction.Normalize();

			f_VecSmoothPath[point] = v3_Direction.x;
			f_VecSmoothPath[point + 1] = v3_Direction.y;

			s_Cadena += f_VecSmoothPath[point];
			s_Cadena += "f,";
			s_Cadena += f_VecSmoothPath[point+1];
			s_Cadena += "f,";
		}
		print (s_Cadena);
	}

	void SaveNormalizedPath() {

		l_TrainingSetIn.Add (f_VecSmoothPath);
		l_CountPerGesture [i_GestureCount - 1]++;
	}

	public void OnAddGesture () {
		
		e_Mode = Mode.ADDGESTURE;
		t_State.text = "ADD GESTURE";
		i_GestureCount++;
		l_CountPerGesture.Add (0);
	}

	// Entrena la red con el training set actual.
	public void OnTrainNetwork () {

		if (l_TrainingSetIn.Count == 0) return;

		CreateTargetOutput ();
		ANN.i_NumOutputs = i_GestureCount;
		ANN.CreateNet ();
		e_Mode = Mode.TRAINING;
		t_State.text = "TRAINING";

		if (ANN.Train(l_TrainingSetIn, l_TargetSetOut)) {
			e_Mode = Mode.ACTIVE;
			t_State.text = "ACTIVE";
		}
		else 
		{
			e_Mode = Mode.ERROR;
			t_State.text = "ERROR";
		}
	}

	public void OnRecognize () {

		if (!ANN.Trained()) return;

		e_Mode = Mode.RECOGNIZE;
		t_State.text = "RECOGNIZE";
	}

	public void OnInsertGestureName () {

		l_GestureName.Add (t_GestureName.text);
		go_PanelGestureName.SetActive (false);
	}

	void CreateTargetOutput () {

		l_TargetSetOut = new List<float[]> ();
		for (int gesture = 0; gesture < l_CountPerGesture.Count; gesture++) {

			for (int gesturecount = 0; gesturecount < l_CountPerGesture[gesture]; gesturecount++) {

				float[] f_Output = new float[i_GestureCount];
				for (int index = 0; index < f_Output.Length; index++) {
					if (index == gesture)
						f_Output[index] = 1;
					else f_Output[index] = 0;
				}
				l_TargetSetOut.Add(f_Output);
			}

		}
	}

	bool Smooth () {

		if (l_VecPath.Count < i_NumSmoothPoints)
			return false;

		l_VecSmoothPath = l_VecPath;

		while (l_VecSmoothPath.Count > i_NumSmoothPoints) {

			float f_ShortestSoFar = Mathf.Infinity;
			int i_PointMarker = 0;

			for (int spanFront = 2; spanFront < l_VecSmoothPath.Count - 1; spanFront++) {

				float f_Length = Mathf.Sqrt (

					(l_VecSmoothPath[spanFront-1].x - l_VecSmoothPath[spanFront].x) * 
					(l_VecSmoothPath[spanFront-1].x - l_VecSmoothPath[spanFront].x) +

					(l_VecSmoothPath[spanFront-1].y - l_VecSmoothPath[spanFront].y) * 
					(l_VecSmoothPath[spanFront-1].y - l_VecSmoothPath[spanFront].y)

					);

				if (f_Length < f_ShortestSoFar) {

					f_ShortestSoFar = f_Length;
					i_PointMarker = spanFront;
				}
			}

			Vector3 newPoint;

			newPoint.x = (l_VecSmoothPath[i_PointMarker - 1].x + 
			              l_VecSmoothPath[i_PointMarker].x) / 2;

			newPoint.y = (l_VecSmoothPath[i_PointMarker - 1].y + 
			              l_VecSmoothPath[i_PointMarker].y) / 2;

			newPoint.z = 0f;

			l_VecSmoothPath[i_PointMarker - 1] = newPoint;
			l_VecSmoothPath.RemoveAt(0 + i_PointMarker);
		}

		return true;
	}



}

