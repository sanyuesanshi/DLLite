#ifndef LAYER_H
#define LAYER_H

#include <memory>
#include <vector>
#include "dl_utility.h"

enum ACT { NON, SIGMOID, RELU, SOFTMAX, CE };
enum KIND { INPUT, HIDEN, LOSS };

class Layer
{
public:
	Layer(int h, ACT act, KIND k = HIDEN);
	int getHeight() const { return m_height; }
	KIND getKind() const { return m_kind; }

	void setBatchSize(int bs);
	int getBatchSize() const { return m_batchSize; }

	std::shared_ptr<Tensor> getDatas() const { return m_datas; }
	std::shared_ptr<Tensor> getWeights() const { return m_weights; }
	std::shared_ptr<Tensor> getBias() const { return m_bias; }
	std::shared_ptr<Tensor> getDa() const { return m_da; }

	void forward(Layer *prev);
	void backward(Layer *prev, float step);

public:
	static const bool gpu = true;

private:
	int m_height;
	int m_batchSize;
	ACT m_activator;
	KIND m_kind;
	std::shared_ptr<Tensor> m_bias;
	std::shared_ptr<Tensor> m_weights;
	std::shared_ptr<Tensor> m_datas;
	std::shared_ptr<Tensor> m_da;
	std::function <void(ITensor *pt)> m_actFunc;
	std::function <void(ITensor* pt)> m_dFunc;
};

class solution
{
public:
	void addLayer(Layer *l) { m_layers.push_back(l); }
	void setBatchSize(int bs)
	{
		for (auto i : m_layers)
		{
			i->setBatchSize(bs);
		}
	}
	void forward() 
	{
		for (auto it = m_layers.begin(); it < m_layers.end(); ++it)
		{
			Layer *prev = (*it)->getKind() == INPUT ? 0 : *(it - 1);
			(*it)->forward(prev);
		}
	};
	void backward(float step)
	{
		for (auto it = m_layers.rbegin(); it < m_layers.rend(); ++it)
		{
			Layer *prev = (*it)->getKind() == INPUT ? 0 : *(it + 1);
			(*it)->backward(prev, step);
		}
	};
private:
	std::vector<Layer*> m_layers;
};
#endif
