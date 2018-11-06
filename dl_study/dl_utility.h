#include "stdafx.h"

#ifndef DL_UTILITY_H
#define DL_UTILITY_H

#include <memory>
#include <vector>
#include <iostream>
#include <functional>
#include <assert.h>

class ITensor
{
public:
	ITensor() = default;
	ITensor(int w, int h):m_width(w), m_height(h){}
	virtual float Value(int x, int y) const = 0;
	virtual float& Value(int x, int y) = 0;
	int getWidth() const { return m_width; }
	int getHeight() const { return m_height; }

	virtual ~ITensor(){}
private:
	int m_width;
	int m_height;
};

class Tensor : public ITensor
{
public:
	Tensor() = default;
	Tensor(int w, int h, std::initializer_list<float> l = {}) : ITensor(w,h), m_values(w * h), m_hDevice(nullptr)
	{
		std::copy(l.begin(), l.end(), m_values.begin());
	}
	float *getDeviceMemory() const { return m_hDevice; }
	//column major
	float Value(int x, int y) const { return m_values[x*getHeight() + y]; }
	float& Value(int x, int y) { return m_values[x*getHeight() + y]; }
	void allocDevice();
	void toDevice();
	void fromDevice();
	~Tensor();
private:
	std::vector<float>   m_values;
	float *m_hDevice;
};

class NormalizeTensor : public ITensor
{
public:
	NormalizeTensor(int w, int h, float v):ITensor(w, h), m_value(v){}
	virtual float Value(int x, int y) const { return m_value; }
	virtual float& Value(int x, int y) { return m_value; }

private:
	float m_value;
};

class Transpose : public ITensor
{
public:
	Transpose(ITensor *t) : ITensor(t->getHeight(), t->getWidth()), m_tensor(t) {}

	float Value(int x, int y) const { return m_tensor->Value(y, x); }
	float& Value(int x, int y) { return m_tensor->Value(y, x); }
private:
	ITensor *m_tensor;
};

class SubTensor : public ITensor
{
public:
	SubTensor(ITensor *t, int offsetX, int offsetY, int w, int h) :
		ITensor(w, h), m_tensor(t), m_offsetX(offsetX), m_offsetY(offsetY){};

	float Value(int x, int y)const { return m_tensor->Value(m_offsetX +x, m_offsetY + y); }
	float& Value(int x, int y) { return m_tensor->Value(m_offsetX + x, m_offsetY + y); }

private:
	ITensor *m_tensor;
	int m_offsetX;
	int m_offsetY;
};

template<typename T1, typename T2, typename F>
void foreach(T1 &r, T2 &l, F &f)
{
	assert(l.getHeight() == r.getHeight() && l.getWidth() == r.getWidth());

	for (int i = 0; i < r.getWidth(); ++i)
	{
		for (int j = 0; j < r.getHeight(); ++j)
		{
			f(r.Value(i, j), l.Value(i, j));
		}
	}
}

template<typename T, typename F>
void foreach(T &r, F &f)
{
	for (int i = 0; i < r.getWidth(); ++i)
	{
		for (int j = 0; j < r.getHeight(); ++j)
		{
			f(r.Value(i, j));
		}
	}
}

float sum(const ITensor &t);
Tensor multipy(const ITensor &l, const ITensor &r);

Tensor op(const ITensor &l, const ITensor &r, std::function<float(float, float)> f);

inline Tensor add(const ITensor &l, const ITensor &r)
{
	return op(l, r, [](float x, float y) {return x + y; });
}

inline Tensor sub(const ITensor &l, const ITensor &r)
{
	return op(l, r, [](float x, float y) {return x - y; });
}

inline Tensor dot(const ITensor &l, const ITensor &r)
{
	return op(l, r, [](float x, float y) {return x * y; });
}

inline Tensor dev(const ITensor &l, const ITensor &r)
{
	return op(l, r, [](float x, float y) {return x / y; });
}

inline void relu(ITensor *pt) {
	foreach(*pt, [](float &x) {x = x > 0 ? x : 0; x = x > 6 ? 6 : x; });
}

inline void sigmoid(ITensor *pt) {
	foreach(*pt, [](float &x) {x = 1.0f / (1.0f + exp(-x)); });
}

inline void drelu(ITensor *pt)
{
	foreach(*pt, [](float &x) {x = (x > 0 && x < 6) ? 1.0f : 0.0f; });
}

inline void dsigmoid(ITensor *pt)
{
	foreach(*pt, [](float &x) {x = x * (1 - x); });
}
void softmax(ITensor *pt);



float conv(const ITensor &image, const ITensor &filter, float b);
float maxPooling(const ITensor &image);
float averagePooling(const ITensor &image);

Tensor processImage(const ITensor &image, const ITensor &filter, std::function<float(const ITensor&)> f, int stride = 1);

Tensor convImage(const ITensor &image, const ITensor &filter, float b, int stride = 1);


float crossEntropy(const ITensor &src, const ITensor &dest);

#endif