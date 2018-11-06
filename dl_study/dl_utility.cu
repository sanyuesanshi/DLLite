#include "stdafx.h"

#include <math.h>
#include <assert.h>
#include <iostream>

#include "dl_utility.h"
#include <cuda_runtime.h>

void Tensor::allocDevice()
{
	if (m_hDevice == nullptr)
	{
		auto error = cudaMalloc(&m_hDevice, m_values.size() * sizeof(float));

		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device vector A (error code %s %lld)!\n", cudaGetErrorString(error), m_values.size());
			
			//exit(EXIT_FAILURE);
		}
	}
}

void Tensor::toDevice()
{
	if (m_hDevice == nullptr) throw "Device memory hasn't allocated yet!";
	auto rc = cudaMemcpy(m_hDevice, &m_values[0], m_values.size() * sizeof(float), cudaMemcpyHostToDevice);
	if (rc != cudaSuccess)
	{
		fprintf(stderr, "Failed %s!\n", cudaGetErrorString(rc));

		//exit(EXIT_FAILURE);
	}
}

void Tensor::fromDevice()
{
	if (m_hDevice == nullptr) throw "Device memory hasn't allocated yet!";
	auto rc = cudaMemcpy(&m_values[0], m_hDevice, m_values.size() * sizeof(float), cudaMemcpyDeviceToHost);
	if (rc != cudaSuccess)
	{
		fprintf(stderr, "Failed %s!\n", cudaGetErrorString(rc));
	}
}
Tensor::~Tensor()
{
	if (m_hDevice)
		cudaFree(m_hDevice);
}

float sum(const ITensor &t)
{
	float result = 0.0f;
	foreach(t, [&result](float t) {result += t; });
	return result;
}

float conv(const ITensor &image, const ITensor &filter, float b)
{
	foreach(image, filter, [&b](float x, float y) {b += x * y; });
	return b;
}

float maxPooling(const ITensor &image)
{
	float result = image.Value(0 , 0);
	foreach(image, [&result](float x) {result = x < result ? result : x; });
	return result;
}

float averagePooling(const ITensor &image)
{
	return sum(image) / image.getWidth() * image.getHeight();
}

void softmax(ITensor *pt)
{
	foreach(*pt, [](float &x) {x = exp(x); });
	float temp = sum(*pt);
	foreach(*pt, [temp](float &x) {x /= temp; });
}

float crossEntropy(const ITensor &src, const ITensor &dest)
{
	float result = 0.0f;
	foreach(src, dest, [&result](float x, float y) {result += y * log(x); });
	return -1 * result;
}

Tensor multipy(const ITensor &l, const ITensor &r)
{
	assert(l.getWidth() == r.getHeight());

	Tensor result( r.getWidth(), l.getHeight());

		for (int k = 0; k < l.getWidth(); ++k)
		{
			for (int i = 0; i < r.getWidth(); ++i)
			{
				for (int j = 0; j < l.getHeight(); ++j)
				{
					result.Value(i, j) +=  r.Value(i, k) * l.Value(k, j);
				}
			}
		}
	
	return result;
}

Tensor convImage(const ITensor &image, const ITensor &filter, float b, int stride)
{
	auto f = [&](const ITensor &t) {return conv(t, filter, b);};
	return processImage(image, filter, f, stride);
}

Tensor op(const ITensor &l, const ITensor &r, std::function<float(float, float)> f)
{
	assert(l.getHeight() == r.getHeight() && l.getWidth() == r.getWidth());
	Tensor result(l.getWidth(), l.getHeight());
	for (int i = 0; i < l.getWidth(); ++i)
		for (int j = 0; j < l.getHeight(); ++j)
		{
			result.Value(i, j) = f(l.Value(i, j), r.Value(i, j));
		}
	return result;
}

Tensor processImage(const ITensor &image, const ITensor &filter, std::function<float(const ITensor&)> f, int stride)
{
	int outWidth = (image.getWidth() - filter.getWidth()) / stride + 1;
	int outHeight = (image.getHeight() - filter.getHeight()) / stride + 1;
	Tensor output(outWidth, outHeight);

	for (int i = 0; i <= image.getWidth() - filter.getWidth(); i += stride)
	{
		for (int j = 0; j <= image.getHeight() - filter.getHeight(); j += stride)
		{
			const SubTensor temp(const_cast<ITensor*>(&image), i, j, filter.getWidth(), filter.getHeight());
			float v = f(temp);
			output.Value(i, j) = v;
		}
	}
	return output;
}