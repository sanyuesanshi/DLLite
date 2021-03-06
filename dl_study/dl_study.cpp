#include "stdafx.h"

#include <vector>
#include <iostream>
#include <numeric>
#include <chrono>
#include <assert.h>
#include <fstream>
#include <utility>
#include <cublas_v2.h>

#include "dl_utility.h"
#include "layer.h"

using namespace std;


cublasHandle_t getCublasHandle()
{
	static cublasHandle_t handle;
	if (handle == 0)
	{
		cublasCreate(&handle);
	}
	return handle;
}


//read data to Tensor
void readData(ifstream &file, ITensor *pt)
{
	foreach(*pt, [&file](float &x) {unsigned char t;  file.read((char*)&t, 1);  x = t;});
}

char readTag(ifstream &file, ITensor *pt)
{
	unsigned char t;
	file.read((char*)&t, 1);
	foreach(*pt, [](float &x) {x = 0.0f; });
	pt->Value(0, t) = 1.0f;
	return t;
}


void NN()
{

	//net topology, 784 notes
	const int N = 28 * 28;
	const int L0 = 300;
	//weight for l1: 300 * 784 matrix
	const int L1 = 300;
	
	const int L2 = 10;

	Layer inputLayer(N, NON, INPUT);
	Layer l0(L0, RELU);
	Layer l1(L1, RELU);// SIGMOID);
	Layer l2(L2, SOFTMAX);
	Layer lossLayer(L2, CE, LOSS);
	solution s;
	s.addLayer(&inputLayer);
	s.addLayer(&l0);
	s.addLayer(&l1);
	s.addLayer(&l2);
	s.addLayer(&lossLayer);
	
	ifstream imagefile;
	ifstream tagfile;
	int magic, numbers, row, clu;
	float step = 0.04f;

	
	
	//trainning.
	for (int epoch = 0; epoch < 100; ++epoch)
	{
		if (epoch >= 10)
			step = 0.02f;
		if (epoch >= 40)
			step = 0.01f;

		imagefile.open("..\\train-images.idx3-ubyte", ios_base::binary | ios_base::in);	
		imagefile.read((char*)&magic, 4).read((char*)&numbers, 4).read((char*)&row, 4).read((char*)&clu, 4);
		tagfile.open("..\\train-labels.idx1-ubyte", ios_base::binary);
		tagfile.read((char*)&magic, 4).read((char*)&numbers, 4);
		cout << magic << ":" << numbers << ":" << row << ":" << clu << endl;

		int trainSize = 60000;
		int batchSize = 1000;

		s.setBatchSize(batchSize);

	
		auto inputs = inputLayer.getDatas();
		auto tags = lossLayer.getDa();
		
		for (int batch = 0; batch < trainSize / batchSize; ++batch)
		{
			//forward
			for (int item = 0; item < batchSize; ++item) 
			{
				SubTensor input(inputs.get(), item, 0, 1, inputs->getHeight());
				SubTensor tag(tags.get(), item, 0, 1, tags->getHeight());
				readData(imagefile, &input);
				readTag(tagfile, &tag);
			}

			s.forward();

			std::cout << "epoch:" << epoch << " batch:" << batch << " loss: " << lossLayer.getDatas()->Value(0,0) << endl;

			//backward
			s.backward(step);
		}
		
		imagefile.close();
		tagfile.close();
	}

	//inferring.
	imagefile.open("..\\t10k-images.idx3-ubyte", ios_base::binary | ios_base::in);
	imagefile.read((char*)&magic, 4).read((char*)&numbers, 4).read((char*)&row, 4).read((char*)&clu, 4);
	cout << "image:" << magic << "," << numbers << "," << row << "," << clu << endl;

	tagfile.open("..\\t10k-labels.idx1-ubyte", ios_base::binary);
	tagfile.read((char*)&magic, 4).read((char*)&numbers, 4);
	cout << "tag:" << magic << "," << numbers << endl;

	int correct = 0;
	int total = 10000;
	int batchSize = 5000;
	s.setBatchSize(batchSize);

	auto inputs = inputLayer.getDatas();
	auto tags = lossLayer.getDa();
	
	for (int batch = 0; batch < total/batchSize; ++batch)
	{
		vector<char> y(batchSize);

		for (int item = 0; item < batchSize; ++item)
		{
			SubTensor input(inputs.get(), item, 0, 1, inputs->getHeight());
			SubTensor tag(tags.get(), item, 0, 1, tags->getHeight());
			readData(imagefile, &input);
			y[item] = readTag(tagfile, &tag);
		}

		//forward		
		s.forward();
		std::cout << " batch:" << batch << " loss: " << lossLayer.getDatas()->Value(0, 0) << endl;
		auto pd = l2.getDatas();

		for (int item = 0; item < batchSize; ++item)
		{
			int index = 0;
			for (int i = 0; i < pd->getHeight(); ++i)
			{
				if (pd->Value(item, index) < pd->Value(item, i))
				{
					index = i;
				}
			}

			if (index == y[item])
			{
				correct++;
			}
		}

		cout << "correct :" << correct << endl;
	}

	if (Layer::gpu)
	{
		cublasDestroy(getCublasHandle());
	}
 }

 int main()
 {
	 NN();
	 getchar();
	 return 0;
 }