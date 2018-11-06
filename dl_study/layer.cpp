#include "stdafx.h"

#include <iostream>
#include <utility>
#include <cublas_v2.h>

#include "layer.h"

extern cublasHandle_t getCublasHandle();

Layer::Layer(int h, ACT act, KIND k):m_height(h),m_batchSize(0), m_activator(act), m_kind(k)
{
	
	switch (act)
	{
	case SIGMOID:
		m_actFunc = sigmoid;
		m_dFunc = dsigmoid;
		break;
	case RELU:
		m_actFunc = relu;
		m_dFunc = drelu;
		break;
	case SOFTMAX:
		m_actFunc = softmax;
		break;
	default:
		;
	};
	
}


void Layer::forward(Layer *prev)
{
	if (prev)
	{
		if (m_kind == LOSS)
		{
			float loss = crossEntropy(*prev->getDatas(), *m_da);
			loss /= m_batchSize;
			m_datas.reset(new Tensor(1, 1, { loss }));
		}
		else
		{
			if (m_weights == nullptr)
			{
				m_weights = std::make_shared<Tensor>(prev->getHeight(), m_height);
				int h = prev->getHeight();
				foreach(*m_weights, [h](float &x) {int t = rand(); t %= (2 * h); t -= h; x = 1.0f / (t ? t : t + 1); });
				m_bias = std::make_shared<Tensor>(1, m_height);

				if (gpu)
				{
					m_weights->allocDevice();
					m_weights->toDevice();
					m_bias->allocDevice();
					m_bias->toDevice();
				}
			}
			if (gpu)
			{
				m_datas->allocDevice();
				float alpha = 1.0f;
				float beta = 0;
				auto rc = cublasSgemm(getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N,
					getWeights()->getHeight(), prev->getDatas()->getWidth(), getWeights()->getWidth(),
					&alpha,
					getWeights()->getDeviceMemory(), getWeights()->getHeight(),
					prev->getDatas()->getDeviceMemory(), prev->getDatas()->getHeight(),
					&beta,
					m_datas->getDeviceMemory(), m_datas->getHeight());

				if (rc != cudaSuccess)
				{
					throw "bad cuda op!";
				}
				
				Tensor base(1, m_batchSize);
				foreach(base, [](float &x) {x = 1.0f; });
				base.allocDevice();
				base.toDevice();
				rc = cublasSger(getCublasHandle(), m_datas->getHeight(), m_datas->getWidth(),
					&alpha,
					m_bias->getDeviceMemory(), 1,
					base.getDeviceMemory(), 1,
					m_datas->getDeviceMemory(), m_datas->getHeight());

				m_datas->fromDevice();
			}
			else
			{
				*(m_datas) = multipy(*m_weights, *prev->getDatas());
				for (int item = 0; item < m_batchSize; ++item)
				{
					SubTensor col(m_datas.get(), item, 0, 1, m_datas->getHeight());
					foreach(col, *m_bias, [](float &x, float y) {x += y; });
				}
			}

			for (int item = 0; item < m_batchSize; ++item)
			{
				SubTensor col(m_datas.get(), item, 0, 1, m_datas->getHeight());
				if (m_actFunc)
					m_actFunc(&col);
			}
		}
	}

	if (gpu && m_kind != LOSS)
	{
		m_datas->allocDevice();
		m_datas->toDevice();
	}
}

void Layer::backward(Layer *prev, float step){
	if (m_kind == INPUT) return;
	if (m_kind == LOSS)
	{
		if (m_activator == CE && prev->m_activator == SOFTMAX)
		{
			*prev->getDa() = sub(*prev->getDatas(), *m_da);
		}
		else
		{
			throw "cannot support non CE loss and non SOFTMAX ouput!";
		}
	}
	else
	{

		step /= m_batchSize;
		Tensor dz;
		if (m_activator == SOFTMAX)
		{
			//only support SOFTMAX with CE loss
			dz = std::move(*m_da);
		}
		else
		{
			assert(m_dFunc);
			m_dFunc(m_datas.get());
			dz = dot(*m_da, *m_datas);
		}
		
		Tensor base(1, m_batchSize);
		foreach(base, [](float &x) {x = 1.0f; });
		Tensor db(1, m_height);
		Tensor dw(prev->getHeight(), m_height);
		auto da = prev->getDa();

		if (gpu)
		{
			dz.allocDevice();
			dz.toDevice();
			base.allocDevice();
			base.toDevice();
			db.allocDevice();
			dw.allocDevice();
			
			if (da)
			{
				float alpha = 1.0f;
				float beta = 0;
				da->allocDevice();
				auto rc = cublasSgemm(getCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
					prev->getHeight(), m_batchSize, m_height,
					&alpha,
					getWeights()->getDeviceMemory(), m_height,
					dz.getDeviceMemory(), dz.getHeight(),
					&beta,
					da->getDeviceMemory(), da->getHeight());

				if (rc != cudaSuccess)
				{
					throw "bad cuda op!";
				}

				da->fromDevice();
			}

			float alpha = -1 * step;
			float beta = 1.0f;
			auto rc = cublasSgemv(getCublasHandle(), CUBLAS_OP_N,
				m_height, m_batchSize,
				&alpha,
				dz.getDeviceMemory(), m_height,
				base.getDeviceMemory(), 1,
				&beta,
				getBias()->getDeviceMemory(), 1);
			if (rc != cudaSuccess)
			{
				throw "bad cuda op!";
			}

			rc = cublasSgemm(getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_T,
				m_height, prev->getHeight(), m_batchSize,
				&alpha,
				dz.getDeviceMemory(), m_height,
				prev->getDatas()->getDeviceMemory(), prev->getHeight(),
				&beta,
				getWeights()->getDeviceMemory(), m_height);

			if (rc != cudaSuccess)
			{
				throw "bad cuda op!";
			}
		}
		else
		{
			if (da)
			{
				*da = multipy(Transpose(getWeights().get()), dz);
			}

			db = multipy(dz, base);
			foreach(db, [step](float &x) {x = step * x; });
			foreach(*getBias().get(), db, [](float &x, float y) {x -= y; });

			dw = multipy(dz, Transpose(prev->getDatas().get()));
			foreach(dw, [step](float &x) {x = step * x; });
			foreach(*getWeights().get(), dw, [](float &x, float y) {return x -= y; });
		}
	}
}

void Layer::setBatchSize(int bs)
{
	if (bs != m_batchSize)
	{
		m_batchSize = bs;
		m_datas = std::make_shared<Tensor>(m_batchSize, m_height);
		if (m_kind != INPUT)
		{
			m_da = std::make_shared<Tensor>(m_batchSize, m_height);
		}
	}
}